/*******************************************************************************
* INTEL CONFIDENTIAL
* Copyright 1996-2023 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
 * Copyright (c) 2005-2021, NumPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials provided
 *        with the distribution.
 *
 *     * Neither the name of the NumPy Developers nor the names of any
 *        contributors may be used to endorse or promote products derived
 *        from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Scalar C99 transcription of __svml_exp8_ha (svml_z0_exp_d_ha.s) and its
 * callout __svml_dexp_ha_cout_rare_internal, one lane at a time. The
 * assembly is the float64 np.exp loop NumPy 2.5.4 runs on AVX-512 hosts,
 * from numpy/_core/src/umath/svml, which NumPy distributes under the
 * BSD-3-Clause licence above. optbench_exp must return its bits.
 *
 * Main path, |x| < 0x1.61da04cbafe44p+9 or x NaN:
 *   T = RZ(x*log2(e) + 1.5*2^48 + 1023)  -> N = T - shifter, a multiple of 1/16
 *   j = low 4 bits of T, Th = 2^(j/16), Tlr = low part of 2^(j/16) relative to Th
 *   r = x - N*ln2 (two steps), |r| forced below 2 by clearing exponent bit 62
 *   exp(x) = 2^floor(N) * Th * (1 + Tlr + r*poly(r))
 * Every packed FMA of the vector code is one fma() call here, with the
 * multiplicands in the order the instruction multiplies them; the one with
 * {rz-sae} rounding is emulated exactly by fma_rz_shifter() with two
 * round-to-nearest fma() calls. All other operations are plain double
 * operations in the original order. The code relies on double arithmetic
 * being IEEE binary64 with round-to-nearest and no excess precision, and on
 * fma() being correctly rounded.
 *
 * Remaining lanes go through the scalar callout, which only uses
 * add/sub/mul/compare and integer exponent manipulation.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "svml_exp.h"

/* __svml_dexp_ha_data_internal_avx512, byte offsets 0..127 (two 8-entry halves). */
static const uint64_t dexp_th[16] = {
    UINT64_C(0x3ff0000000000000), UINT64_C(0x3ff0b5586cf9890f),
    UINT64_C(0x3ff172b83c7d517b), UINT64_C(0x3ff2387a6e756238),
    UINT64_C(0x3ff306fe0a31b715), UINT64_C(0x3ff3dea64c123422),
    UINT64_C(0x3ff4bfdad5362a27), UINT64_C(0x3ff5ab07dd485429),
    UINT64_C(0x3ff6a09e667f3bcd), UINT64_C(0x3ff7a11473eb0187),
    UINT64_C(0x3ff8ace5422aa0db), UINT64_C(0x3ff9c49182a3f090),
    UINT64_C(0x3ffae89f995ad3ad), UINT64_C(0x3ffc199bdd85529c),
    UINT64_C(0x3ffd5818dcfba487), UINT64_C(0x3ffea4afa2a490da)
};

/* Byte offsets 128..255. */
static const uint64_t dexp_tlr[16] = {
    UINT64_C(0x0000000000000000), UINT64_C(0x3c979aa65d837b6d),
    UINT64_C(0xbc801b15eaa59348), UINT64_C(0x3c968efde3a8a894),
    UINT64_C(0x3c834d754db0abb6), UINT64_C(0x3c859f48a72a4c6d),
    UINT64_C(0x3c7690cebb7aafb0), UINT64_C(0x3c9063e1e21c5409),
    UINT64_C(0xbc93b3efbf5e2228), UINT64_C(0xbc7b32dcb94da51d),
    UINT64_C(0x3c8db72fc1f0eab4), UINT64_C(0x3c71affc2b91ce27),
    UINT64_C(0x3c8c1a7792cb3387), UINT64_C(0x3c736eae30af0cb3),
    UINT64_C(0x3c74a385a63d07a7), UINT64_C(0xbc8ff7128fd391f0)
};

/* Broadcast constants; the comment is the byte offset of the 64-byte block. */
#define DEXP_L2E     UINT64_C(0x3ff71547652b82fe) /*  256: log2(e) */
#define DEXP_SHIFTER UINT64_C(0x42f8000000003ff0) /*  320: 1.5*2^48 + 1023 */
#define DEXP_LN2HI   UINT64_C(0x3fe62e42fefa39ef) /*  384 */
#define DEXP_LN2LO   UINT64_C(0x3c7abc9e3b39803f) /*  448 */
#define DEXP_RMASK   UINT64_C(0xbfffffffffffffff) /*  512 */
#define DEXP_C6      UINT64_C(0x3f57411836940c04) /*  576 */
#define DEXP_C5      UINT64_C(0x3f81101cbbc265c0) /*  640 */
#define DEXP_C4      UINT64_C(0x3fa55557242d68fe) /*  704 */
#define DEXP_C3      UINT64_C(0x3fc5555553939732) /*  768 */
#define DEXP_C2      UINT64_C(0x3fe000000000d008) /*  832 */
#define DEXP_C1      UINT64_C(0x3fefffffffffff70) /*  896 */
#define DEXP_ABSMASK UINT64_C(0x7fffffffffffffff) /*  960 */
#define DEXP_THRESH  UINT64_C(0x40861da04cbafe44) /* 1024 */

/* _imldExpHATab, byte offsets 0..1023: {2^(j/64) high, low} for j = 0..63. */
static const uint64_t imld_exp_tab[128] = {
    UINT64_C(0x3ff0000000000000), UINT64_C(0x0000000000000000),
    UINT64_C(0x3ff02c9a3e778061), UINT64_C(0xbc7160139cd8dc5d),
    UINT64_C(0x3ff059b0d3158574), UINT64_C(0x3c8cd2523567f613),
    UINT64_C(0x3ff0874518759bc8), UINT64_C(0x3c60f74e61e6c861),
    UINT64_C(0x3ff0b5586cf9890f), UINT64_C(0x3c979aa65d837b6d),
    UINT64_C(0x3ff0e3ec32d3d1a2), UINT64_C(0x3c3ebe3d702f9cd1),
    UINT64_C(0x3ff11301d0125b51), UINT64_C(0xbc9556522a2fbd0e),
    UINT64_C(0x3ff1429aaea92de0), UINT64_C(0xbc91c923b9d5f416),
    UINT64_C(0x3ff172b83c7d517b), UINT64_C(0xbc801b15eaa59348),
    UINT64_C(0x3ff1a35beb6fcb75), UINT64_C(0x3c8b898c3f1353bf),
    UINT64_C(0x3ff1d4873168b9aa), UINT64_C(0x3c9aecf73e3a2f60),
    UINT64_C(0x3ff2063b88628cd6), UINT64_C(0x3c8a6f4144a6c38d),
    UINT64_C(0x3ff2387a6e756238), UINT64_C(0x3c968efde3a8a894),
    UINT64_C(0x3ff26b4565e27cdd), UINT64_C(0x3c80472b981fe7f2),
    UINT64_C(0x3ff29e9df51fdee1), UINT64_C(0x3c82f7e16d09ab31),
    UINT64_C(0x3ff2d285a6e4030b), UINT64_C(0x3c8b3782720c0ab4),
    UINT64_C(0x3ff306fe0a31b715), UINT64_C(0x3c834d754db0abb6),
    UINT64_C(0x3ff33c08b26416ff), UINT64_C(0x3c8fdd395dd3f84a),
    UINT64_C(0x3ff371a7373aa9cb), UINT64_C(0xbc924aedcc4b5068),
    UINT64_C(0x3ff3a7db34e59ff7), UINT64_C(0xbc71d1e83e9436d2),
    UINT64_C(0x3ff3dea64c123422), UINT64_C(0x3c859f48a72a4c6d),
    UINT64_C(0x3ff4160a21f72e2a), UINT64_C(0xbc58a78f4817895b),
    UINT64_C(0x3ff44e086061892d), UINT64_C(0x3c4363ed60c2ac11),
    UINT64_C(0x3ff486a2b5c13cd0), UINT64_C(0x3c6ecce1daa10379),
    UINT64_C(0x3ff4bfdad5362a27), UINT64_C(0x3c7690cebb7aafb0),
    UINT64_C(0x3ff4f9b2769d2ca7), UINT64_C(0xbc8f94340071a38e),
    UINT64_C(0x3ff5342b569d4f82), UINT64_C(0xbc78dec6bd0f385f),
    UINT64_C(0x3ff56f4736b527da), UINT64_C(0x3c93350518fdd78e),
    UINT64_C(0x3ff5ab07dd485429), UINT64_C(0x3c9063e1e21c5409),
    UINT64_C(0x3ff5e76f15ad2148), UINT64_C(0x3c9432e62b64c035),
    UINT64_C(0x3ff6247eb03a5585), UINT64_C(0xbc8c33c53bef4da8),
    UINT64_C(0x3ff6623882552225), UINT64_C(0xbc93cedd78565858),
    UINT64_C(0x3ff6a09e667f3bcd), UINT64_C(0xbc93b3efbf5e2228),
    UINT64_C(0x3ff6dfb23c651a2f), UINT64_C(0xbc6367efb86da9ee),
    UINT64_C(0x3ff71f75e8ec5f74), UINT64_C(0xbc781f647e5a3ecf),
    UINT64_C(0x3ff75feb564267c9), UINT64_C(0xbc8619321e55e68a),
    UINT64_C(0x3ff7a11473eb0187), UINT64_C(0xbc7b32dcb94da51d),
    UINT64_C(0x3ff7e2f336cf4e62), UINT64_C(0x3c65ebe1abd66c55),
    UINT64_C(0x3ff82589994cce13), UINT64_C(0xbc9369b6f13b3734),
    UINT64_C(0x3ff868d99b4492ed), UINT64_C(0xbc94d450d872576e),
    UINT64_C(0x3ff8ace5422aa0db), UINT64_C(0x3c8db72fc1f0eab4),
    UINT64_C(0x3ff8f1ae99157736), UINT64_C(0x3c7bf68359f35f44),
    UINT64_C(0x3ff93737b0cdc5e5), UINT64_C(0xbc5da9b88b6c1e29),
    UINT64_C(0x3ff97d829fde4e50), UINT64_C(0xbc92434322f4f9aa),
    UINT64_C(0x3ff9c49182a3f090), UINT64_C(0x3c71affc2b91ce27),
    UINT64_C(0x3ffa0c667b5de565), UINT64_C(0xbc87c50422622263),
    UINT64_C(0x3ffa5503b23e255d), UINT64_C(0xbc91bbd1d3bcbb15),
    UINT64_C(0x3ffa9e6b5579fdbf), UINT64_C(0x3c8469846e735ab3),
    UINT64_C(0x3ffae89f995ad3ad), UINT64_C(0x3c8c1a7792cb3387),
    UINT64_C(0x3ffb33a2b84f15fb), UINT64_C(0xbc55c3d956dcaeba),
    UINT64_C(0x3ffb7f76f2fb5e47), UINT64_C(0xbc68d6f438ad9334),
    UINT64_C(0x3ffbcc1e904bc1d2), UINT64_C(0x3c74ffd70a5fddcd),
    UINT64_C(0x3ffc199bdd85529c), UINT64_C(0x3c736eae30af0cb3),
    UINT64_C(0x3ffc67f12e57d14b), UINT64_C(0x3c84e08fd10959ac),
    UINT64_C(0x3ffcb720dcef9069), UINT64_C(0x3c676b2c6c921968),
    UINT64_C(0x3ffd072d4a07897c), UINT64_C(0xbc8fad5d3ffffa6f),
    UINT64_C(0x3ffd5818dcfba487), UINT64_C(0x3c74a385a63d07a7),
    UINT64_C(0x3ffda9e603db3285), UINT64_C(0x3c8e5a50d5c192ac),
    UINT64_C(0x3ffdfc97337b9b5f), UINT64_C(0xbc82d52107b43e1f),
    UINT64_C(0x3ffe502ee78b3ff6), UINT64_C(0x3c74b604603a88d3),
    UINT64_C(0x3ffea4afa2a490da), UINT64_C(0xbc8ff7128fd391f0),
    UINT64_C(0x3ffefa1bee615a27), UINT64_C(0x3c8ec3bc41aa2008),
    UINT64_C(0x3fff50765b6e4540), UINT64_C(0x3c8a64a931d185ee),
    UINT64_C(0x3fffa7c1819e90d8), UINT64_C(0x3c77893b4d91cd9d)
};

/* _imldExpHATab, byte offsets 1024..1168. */
#define IMLD_L2E64   UINT64_C(0x40571547652b82fe) /* 1024: 64*log2(e) */
#define IMLD_SHIFTER UINT64_C(0x4338000000000000) /* 1032: 1.5*2^52 */
#define IMLD_A1      UINT64_C(0x3fe0000000000000) /* 1040 */
#define IMLD_A2      UINT64_C(0x3fc55555555548f8) /* 1048 */
#define IMLD_A3      UINT64_C(0x3fa5555555558fcc) /* 1056 */
#define IMLD_A4      UINT64_C(0x3f8111123aaf20d3) /* 1064 */
#define IMLD_A5      UINT64_C(0x3f56c16a1c2a3ffd) /* 1072 */
#define IMLD_OVF     UINT64_C(0x40862e42fefa39ef) /* 1080 */
#define IMLD_SUBN    UINT64_C(0xc086232bdd7abcd2) /* 1088 */
#define IMLD_UNF     UINT64_C(0xc0874910d52d3051) /* 1096 */
#define IMLD_LN2HI64 UINT64_C(0x3f862e42fefa0000) /* 1104 */
#define IMLD_LN2LO64 UINT64_C(0x3d1cf79abc9e3b3a) /* 1112 */
#define IMLD_TINY    UINT64_C(0x0010000000000001) /* 1120 */
#define IMLD_HUGE    UINT64_C(0x7fefffffffffffff) /* 1128 */
#define IMLD_ZERO    UINT64_C(0x0000000000000000) /* 1136 */
#define IMLD_ONE     UINT64_C(0x3ff0000000000000) /* 1144 */
#define IMLD_TWO     UINT64_C(0x4000000000000000) /* 1152 */
#define IMLD_TWOM60  UINT64_C(0x3c30000000000000) /* 1160 */
#define IMLD_SPLIT   UINT64_C(0x41f8000000000000) /* 1168 */

static double as_double(uint64_t u)
{
    double d;
    memcpy(&d, &u, sizeof d);
    return d;
}

static uint64_t as_bits(double d)
{
    uint64_t u;
    memcpy(&u, &d, sizeof u);
    return u;
}

#define D(name) as_double(name)

/*
 * vfmadd213pd {rz-sae}: RZ(x*l2e + s) for the shifter s = 1.5*2^48 + 1023.
 *
 * Valid whenever |x*l2e| < 2^46 (the main path has |x*l2e| <= 1021): the
 * exact value v = x*l2e + s is then positive and t = RN(v) lies in
 * [s/2, 2s], so s - t is exact (Sterbenz) and v - t = x*l2e + (s - t)
 * exactly. The single rounding of c = fma(x, l2e, s - t) keeps the sign of
 * v - t, because a nonzero v - t is never small enough to round to zero:
 * it is x*l2e itself (magnitude above 2^-1074) when s == t, and otherwise
 * a nonzero multiple of 2^-4 plus x*l2e, which can only cancel when
 * |x| > 2^-6, making it a multiple of 2^-109. RZ(v) is t when v >= t and
 * the next double below t when v < t, because RN and RZ of a positive value
 * differ by at most one ulp. A NaN x makes t NaN and the comparison false,
 * so t is returned unchanged.
 */
static double fma_rz_shifter(double x, double l2e, double s)
{
    double t = fma(x, l2e, s);
    double c = fma(x, l2e, s - t);
    if (c < 0.0)
        t = as_double(as_bits(t) - 1);
    return t;
}

static double quiet(double a)
{
    return as_double(as_bits(a) | UINT64_C(0x0008000000000000));
}

/*
 * vscalefpd {rn-sae}: a * 2^floor(n), NaN in a takes precedence over NaN
 * in n. On the main path a is finite (r has |r| < 2 even for NaN x) and,
 * unless n is NaN, -1021 <= n < 1021, so 2^floor(n) is a normal double and
 * one rounded multiplication gives the same result as the instruction.
 */
static double scalef(double a, double n)
{
    int k;

    if (a != a)
        return quiet(a);
    if (n != n)
        return quiet(n);
    k = (int)floor(n);
    return a * as_double((uint64_t)(k + 1023) << 52);
}

/* Replace the sign/exponent half-word of tmpl like movzwl/andl $-32753/orl/movw. */
static double with_exponent(uint64_t tmpl, uint32_t e)
{
    uint32_t hw = (uint32_t)(tmpl >> 48);
    hw = (hw & UINT32_C(0xffff800f)) | (e << 4);
    return as_double((tmpl & UINT64_C(0x0000ffffffffffff)) | ((uint64_t)(hw & 0xffffu) << 48));
}

/* __svml_dexp_ha_cout_rare_internal. */
static double exp_rare(double x)
{
    uint64_t ix = as_bits(x);
    int ex = (int)((ix >> 48) & 0x7ff0u) >> 4;
    double y, tsh, n, r, p, th, res;
    uint32_t lo, j, m, e;

    if (ex == 2047) {
        if ((ix >> 63) != 0 && ((ix >> 32) & 0xfffffu) == 0 && (uint32_t)ix == 0)
            return D(IMLD_ZERO);
        return x * x;
    }
    if (ex <= 970)
        return D(IMLD_ONE) + x;
    if (!(D(IMLD_OVF) >= x))
        return D(IMLD_HUGE) * D(IMLD_HUGE);
    if (!(x >= D(IMLD_UNF)))
        return D(IMLD_TINY) * D(IMLD_TINY);

    y = D(IMLD_L2E64) * x;
    tsh = y + D(IMLD_SHIFTER);
    lo = (uint32_t)as_bits(tsh);
    j = lo & 63u;
    m = lo >> 6;
    n = tsh - D(IMLD_SHIFTER);
    r = x - n * D(IMLD_LN2HI64);
    r = r - n * D(IMLD_LN2LO64);
    p = D(IMLD_A5) * r;
    p = p + D(IMLD_A4);
    p = p * r;
    p = p + D(IMLD_A3);
    p = p * r;
    p = p + D(IMLD_A2);
    p = p * r;
    p = p + D(IMLD_A1);
    p = p * r;
    p = p * r;
    p = p + r;
    p = p + D(imld_exp_tab[2 * j + 1]);
    th = D(imld_exp_tab[2 * j]);
    p = p * th;

    if (!(x >= D(IMLD_SUBN))) {
        double sc, plo, hi, sum, err, t, shi, slo;

        e = (m + 1083u) & 2047u;
        sc = with_exponent(IMLD_ZERO, e);
        plo = p * sc;
        hi = sc * th;
        sum = hi + plo;
        if (e <= 50u)
            return sum * D(IMLD_TWOM60);
        err = hi - sum;
        err = err + plo;
        t = sum * D(IMLD_SPLIT);
        shi = sum + t;
        shi = shi - t;
        slo = sum - shi;
        slo = err + slo;
        shi = shi * D(IMLD_TWOM60);
        slo = slo * D(IMLD_TWOM60);
        return shi + slo;
    }

    e = (m + 1023u) & 2047u;
    res = p + th;
    if (e > 2046u) {
        e = (e - 1u) & 2047u;
        return res * with_exponent(IMLD_ZERO, e) * D(IMLD_TWO);
    }
    return res * with_exponent(IMLD_ZERO, e);
}

/* One lane of __svml_exp8_ha. */
static double exp8_ha_lane(double x)
{
    double ax, t, n, th, tlr, r, r2, p56, p34, p12, p36, p16, p, res;
    unsigned j;

    ax = as_double(as_bits(x) & DEXP_ABSMASK);
    if (ax >= D(DEXP_THRESH))
        return exp_rare(x);

    t = fma_rz_shifter(x, D(DEXP_L2E), D(DEXP_SHIFTER));
    n = t - D(DEXP_SHIFTER);
    j = (unsigned)(as_bits(t) & 15u);
    tlr = D(dexp_tlr[j]);
    th = D(dexp_th[j]);
    r = fma(-n, D(DEXP_LN2HI), x);
    r = fma(-D(DEXP_LN2LO), n, r);
    r = as_double(as_bits(r) & DEXP_RMASK);
    r2 = r * r;
    p56 = fma(D(DEXP_C6), r, D(DEXP_C5));
    p34 = fma(D(DEXP_C4), r, D(DEXP_C3));
    p12 = fma(D(DEXP_C2), r, D(DEXP_C1));
    p36 = fma(r2, p56, p34);
    p16 = fma(r2, p36, p12);
    p = fma(p16, r, tlr);
    res = fma(th, p, th);
    return scalef(res, n);
}

double optbench_exp(double x)
{
    return exp8_ha_lane(x);
}

void optbench_exp_array(const double *x, double *out, long n)
{
    long i;

    for (i = 0; i < n; i++)
        out[i] = exp8_ha_lane(x[i]);
}
