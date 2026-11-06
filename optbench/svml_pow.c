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
 * License of the SVML sources as distributed with NumPy
 * (numpy/_core/src/umath/svml/LICENSE):
 *
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
 * Scalar C99 transcription of __svml_pow8_ha and its special-case helper
 * __svml_dpow_ha_cout_rare_internal from svml_z0_pow_d_ha.s.
 *
 * Main path: one lane of the AVX-512 code.  Every vfmadd/vfmsub/vfnmadd is
 * an fma() call with the operands in instruction order, every other
 * arithmetic instruction is the same plain operation.  Instructions with
 * embedded rounding ({rz-sae}, {rd-sae}) are evaluated exactly in the
 * default round-to-nearest mode by the *_rz / *_rd helpers below.
 *
 * Rare path: the scalar SSE2 routine, transcribed instruction by
 * instruction; stack slots of the routine are named mNN after their
 * offset -NN(%rsp).
 *
 * optbench_pow_scalar_array() additionally reproduces the fast paths that
 * NumPy's DOUBLE_power loop takes for a scalar exponent before it calls
 * the vector routine.
 *
 * Correctness of the transcription depends on fma() being correctly
 * rounded (true for glibc on every x86-64 host, with or without FMA
 * hardware) and on compiling with -ffp-contract=off and without
 * -ffast-math.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "svml_pow.h"

static inline uint64_t as_u64(double d)
{
    uint64_t u;
    memcpy(&u, &d, sizeof u);
    return u;
}

static inline double as_f64(uint64_t u)
{
    double d;
    memcpy(&d, &u, sizeof d);
    return d;
}

#define F(k) as_f64(k)
#define CT(i) as_f64(pow_cout_tab[(i)])

/* data+0..255: Th, -log2(DblRcp) (+1 for DblRcp>=1.5), 32 entries */
static const uint64_t pow_log_th[32] = {
    0x0000000000000000ULL, 0xbfa6bad3758e0000ULL, 0xbfb663f6fac90000ULL,
    0xbfc08c588cda8000ULL, 0xbfc5c01a39fc0000ULL, 0xbfcacf5e2db50000ULL,
    0xbfcfbc16b9028000ULL, 0xbfd24407ab0e0000ULL, 0xbfd49a784bcd0000ULL,
    0xbfd6e221cd9d0000ULL, 0xbfd91bba891f0000ULL, 0xbfdb47ebf7388000ULL,
    0xbfdd6753e0330000ULL, 0xbfdf7a8568cb0000ULL, 0xbfe0c10500d64000ULL,
    0xbfe1bf311e95e000ULL, 0x3fda8ff971810000ULL, 0x3fd8a8980abfc000ULL,
    0x3fd6cb0f6865c000ULL, 0x3fd4f6fbb2cec000ULL, 0x3fd32bfee3710000ULL,
    0x3fd169c053640000ULL, 0x3fcf5fd8a9060000ULL, 0x3fcbfc67a8000000ULL,
    0x3fc8a8980abf8000ULL, 0x3fc563dc29ff8000ULL, 0x3fc22dadc2ab0000ULL,
    0x3fbe0b1ae8f30000ULL, 0x3fb7d60496d00000ULL, 0x3fb1bb32a6000000ULL,
    0x3fa77394c9da0000ULL, 0x3f9743ee86200000ULL
};

/* data+256..511: Tl, low part of Th */
static const uint64_t pow_log_tl[32] = {
    0x0000000000000000ULL, 0xbd5fb0e626c0de13ULL, 0xbd33167ccc538261ULL,
    0x3d2871a7610e40bdULL, 0x3d54bc302ffa76fbULL, 0x3d436c101ee13440ULL,
    0x3d47f5dc57266758ULL, 0xbd3ce60916e52e91ULL, 0xbd5b8afe492bf6ffULL,
    0xbd49bcaf1aa4168aULL, 0xbd5708b4b2b5056cULL, 0xbd250520a377c7ecULL,
    0x3d55f101c141e670ULL, 0xbd3b3b3864c60011ULL, 0x3d45669df6a2b592ULL,
    0x3d5fe43895d8ac46ULL, 0x3d44bc302ffa76fbULL, 0xbd266cccab240e90ULL,
    0x3d41d406db502403ULL, 0x3d3661e393a16b95ULL, 0xbd51979a5db68722ULL,
    0xbd4d4f1b95e0ff45ULL, 0x3d5f1a4847f7b278ULL, 0xbd3667f21fa8423fULL,
    0x3d5e9933354dbf17ULL, 0x3d56590643906f2aULL, 0x3d5a4b69691d7994ULL,
    0xbd054cda62d3926eULL, 0xbd512ce6312ebb82ULL, 0x3d552743318a8a57ULL,
    0xbd54e55443478fe0ULL, 0xbd495539356f93dcULL
};

/* data+512..639: The = 2^(j/16) */
static const uint64_t pow_exp_the[16] = {
    0x3ff0000000000000ULL, 0x3ff0b5586cf9890fULL, 0x3ff172b83c7d517bULL,
    0x3ff2387a6e756238ULL, 0x3ff306fe0a31b715ULL, 0x3ff3dea64c123422ULL,
    0x3ff4bfdad5362a27ULL, 0x3ff5ab07dd485429ULL, 0x3ff6a09e667f3bcdULL,
    0x3ff7a11473eb0187ULL, 0x3ff8ace5422aa0dbULL, 0x3ff9c49182a3f090ULL,
    0x3ffae89f995ad3adULL, 0x3ffc199bdd85529cULL, 0x3ffd5818dcfba487ULL,
    0x3ffea4afa2a490daULL
};

/* data+640..767: Tle, low part of The divided by The */
static const uint64_t pow_exp_tle[16] = {
    0x0000000000000000ULL, 0x3c979aa65d837b6dULL, 0xbc801b15eaa59348ULL,
    0x3c968efde3a8a894ULL, 0x3c834d754db0abb6ULL, 0x3c859f48a72a4c6dULL,
    0x3c7690cebb7aafb0ULL, 0x3c9063e1e21c5409ULL, 0xbc93b3efbf5e2228ULL,
    0xbc7b32dcb94da51dULL, 0x3c8db72fc1f0eab4ULL, 0x3c71affc2b91ce27ULL,
    0x3c8c1a7792cb3387ULL, 0x3c736eae30af0cb3ULL, 0x3c74a385a63d07a7ULL,
    0xbc8ff7128fd391f0ULL
};

/* broadcast constants of the data table, named by byte offset */
#define K768  0x3fe0000000000000ULL
#define K832  0x3ff8000000000000ULL
#define K896  0x3ff0000000000000ULL
#define K960  0x40071547652b82feULL
#define K1024 0xc0627a394386376fULL
#define K1088 0x4054873cf87141d1ULL
#define K1152 0xc04715473528efc1ULL
#define K1216 0x403a617607e55815ULL
#define K1280 0xc02ec709dc3b71a1ULL
#define K1344 0x4022776c50effbdeULL
#define K1408 0xc0171547652b82fcULL
#define K1472 0x400ec709dc3a03fdULL
#define K1536 0xbc8778d9c7190437ULL
#define K1600 0x3c8777df70b75d10ULL
#define K1664 0x42f8000000003ff0ULL
#define K1728 0xbfffffffffffffffULL
#define K1792 0x3f24a1d7f58c2d59ULL
#define K1856 0x3f55d7472783d279ULL
#define K1920 0x3f83b2ad1b14ebaaULL
#define K1984 0x3fac6b08d4ad8eb9ULL
#define K2048 0x3fcebfbdff84554dULL
#define K2112 0x3fe62e42fefa398bULL
#define K2368 0x7fffffffffffffffULL
#define K2432 0x408fec0000000000ULL
#define K2496 0x7ff0000000000000ULL

/* __dpow_ha_CoutTab, 860 entries (6880 bytes) */
static const uint64_t pow_cout_tab[860] = {
    0x3ff0000000000000ULL, 0x3fef07c000000000ULL, 0x3fee1e0000000000ULL,
    0x3fed41c000000000ULL, 0x3fec71c000000000ULL, 0x3febad0000000000ULL,
    0x3feaf28000000000ULL, 0x3fea41c000000000ULL, 0x3fe9998000000000ULL,
    0x3fe8f9c000000000ULL, 0x3fe8618000000000ULL, 0x3fe7d04000000000ULL,
    0x3fe745c000000000ULL, 0x3fe6c18000000000ULL, 0x3fe642c000000000ULL,
    0x3fe5c98000000000ULL, 0x3fe5554000000000ULL, 0x3fe4e60000000000ULL,
    0x3fe47b0000000000ULL, 0x3fe4140000000000ULL, 0x3fe3b14000000000ULL,
    0x3fe3520000000000ULL, 0x3fe2f68000000000ULL, 0x3fe29e4000000000ULL,
    0x3fe2494000000000ULL, 0x3fe1f70000000000ULL, 0x3fe1a7c000000000ULL,
    0x3fe15b0000000000ULL, 0x3fe1110000000000ULL, 0x3fe0c98000000000ULL,
    0x3fe0844000000000ULL, 0x3fe0410000000000ULL, 0x3fe0000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x3fa6bb01a01f0000ULL,
    0x3d3c995e7439db71ULL, 0x3fb665684ff80000ULL, 0x3d3084f28da93fb0ULL,
    0x3fc08cd7820ec000ULL, 0x3d3543c511b40207ULL, 0x3fc5c04864906000ULL,
    0x3d28c5d47e5f3668ULL, 0x3fcacf30032be000ULL, 0x3d2e3733debf9166ULL,
    0x3fcfbc44e396e000ULL, 0x3d47b3f928665438ULL, 0x3fd243a590d2b000ULL,
    0x3d3c9b75e8e9d45dULL, 0x3fd49ad4a118d000ULL, 0x3d45cd377302cca6ULL,
    0x3fd6e22792ef1000ULL, 0x3d314f240e7e9039ULL, 0x3fd91bd19e695000ULL,
    0x3d4dbb3ee4f6c667ULL, 0x3fdb4865273ed000ULL, 0x3d4ab54a099e1f61ULL,
    0x3fdd679920231000ULL, 0x3d18ed5096e87504ULL, 0x3fdf7a349e747000ULL,
    0x3d4a6e7081d99120ULL, 0x3fe0c11650cf0000ULL, 0x3d461752eb1152a5ULL,
    0x3fe1bf426e8e9800ULL, 0x3d376aff6c055f56ULL, 0xbfda8f9d1c354000ULL,
    0xbd4604f54f4f9854ULL, 0xbfd8a9228a043000ULL, 0xbd49bc20cf8dd884ULL,
    0xbfd6cb99e7aa4000ULL, 0xbd412b5ad5a7002bULL, 0xbfd4f69f5d830000ULL,
    0xbd38f36dd24bae46ULL, 0xbfd32c15f8b57000ULL, 0xbd0d6ee2e01d9232ULL,
    0xbfd16935d34fb000ULL, 0xbd2151c6348d84a5ULL, 0xbfcf5faa7e71a000ULL,
    0xbd3d157620c552c2ULL, 0xbfcbfc5c1d5be000ULL, 0xbd2784900e42b538ULL,
    0xbfc8a9ad0948a000ULL, 0xbd4c89ba64f25a56ULL, 0xbfc563adff6b0000ULL,
    0xbd4d0837079422c3ULL, 0xbfc22df302746000ULL, 0xbd3048e3c2505d3dULL,
    0xbfbe089492ee4000ULL, 0xbd405589fcd57f87ULL, 0xbfb7d49341a08000ULL,
    0xbd4eeef8bcf7aa55ULL, 0xbfb1bc75cf5a4000ULL, 0xbd4d7db2139e8397ULL,
    0xbfa778fdc1828000ULL, 0xbd34378af2af5333ULL, 0xbf97427d317a0000ULL,
    0xbd4700f14b03b094ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x3ff0000000000000ULL, 0x3feff00000000000ULL, 0x3fefe00000000000ULL,
    0x3fefd04000000000ULL, 0x3fefc08000000000ULL, 0x3fefb0c000000000ULL,
    0x3fefa10000000000ULL, 0x3fef918000000000ULL, 0x3fef820000000000ULL,
    0x3fef728000000000ULL, 0x3fef630000000000ULL, 0x3fef53c000000000ULL,
    0x3fef448000000000ULL, 0x3fef354000000000ULL, 0x3fef260000000000ULL,
    0x3ff0454000000000ULL, 0x3ff0410000000000ULL, 0x3ff03d0000000000ULL,
    0x3ff038c000000000ULL, 0x3ff034c000000000ULL, 0x3ff0308000000000ULL,
    0x3ff02c8000000000ULL, 0x3ff0288000000000ULL, 0x3ff0244000000000ULL,
    0x3ff0204000000000ULL, 0x3ff01c4000000000ULL, 0x3ff0184000000000ULL,
    0x3ff0140000000000ULL, 0x3ff0100000000000ULL, 0x3ff00c0000000000ULL,
    0x3ff0080000000000ULL, 0x3ff0040000000000ULL, 0x3ff0000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x3f671b0ea4280000ULL,
    0x3d497f689876ef47ULL, 0x3f7720d9c0680000ULL, 0x3d441af53778c7ccULL,
    0x3f8145c68a7a0000ULL, 0x3d44bdd19ac06488ULL, 0x3f86fdf461d20000ULL,
    0x3d3c9efc46d9f6f7ULL, 0x3f8cb8f9355c0000ULL, 0x3d3275c9adfbb459ULL,
    0x3f913b6bec350000ULL, 0x3d0f25b44fc32adbULL, 0x3f9410167b760000ULL,
    0x3d3f880ca8ed5317ULL, 0x3f96e625317a0000ULL, 0x3d453f0e645614dbULL,
    0x3f99bd996d010000ULL, 0x3d2cd686341a2dabULL, 0x3f9c96748ed00000ULL,
    0x3d4ef88d66d10b04ULL, 0x3f9f64edd4200000ULL, 0x3d4511c3828828daULL,
    0x3fa11a6297920000ULL, 0x3d4d925cd7d436d6ULL, 0x3fa282fdfad70000ULL,
    0x3d49eee0a58b8d6eULL, 0x3fa3ec49bfc20000ULL, 0x3d4deac32e0e0086ULL,
    0xbf98c49331230000ULL, 0xbd49ad075efcabfaULL, 0xbf97427d317a0000ULL,
    0xbd4700f14b03b094ULL, 0xbf95d6c0ee910000ULL, 0xbd4a5115535202a3ULL,
    0xbf9453e6adb60000ULL, 0xbd415a44e102f731ULL, 0xbf92e77157080000ULL,
    0xbd4c7ed85ee9ad86ULL, 0xbf9163d20d100000ULL, 0xbd46e8b9664fe33fULL,
    0xbf8fed45cbcc0000ULL, 0xbd37f33943464056ULL, 0xbf8d1232c5ea0000ULL,
    0xbd17cf34b0bdc8dfULL, 0xbf8a08a828680000ULL, 0xbd35a529f02b9ccfULL,
    0xbf872c1f4cf00000ULL, 0xbd2b4934580fe573ULL, 0xbf844ee0a6f20000ULL,
    0xbd24c8cbff314317ULL, 0xbf8170ebdc1c0000ULL, 0xbd447db02cc5232fULL,
    0xbf7cc89f97d40000ULL, 0xbd43ac9c90330e7bULL, 0xbf7709c46d780000ULL,
    0xbd4563ba56cde925ULL, 0xbf71497accf40000ULL, 0xbd4f08e7ddd3e770ULL,
    0xbf670f83ff080000ULL, 0xbd33ab2631d4676dULL, 0xbf57126537400000ULL,
    0xbd2fa2a0fd4fca1dULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x3ff7154740000000ULL, 0x3ff7151900000000ULL, 0x3ff714eb00000000ULL,
    0x3ff714bcc0000000ULL, 0x3ff7148e80000000ULL, 0x3ff7146080000000ULL,
    0x3ff7143240000000ULL, 0x3ff7140440000000ULL, 0x3ff713d600000000ULL,
    0x3ff713a7c0000000ULL, 0x3ff71379c0000000ULL, 0x3ff7134b80000000ULL,
    0x3ff7131d80000000ULL, 0x3ff712ef40000000ULL, 0x3ff712c140000000ULL,
    0x3ff7129300000000ULL, 0x3ff7126500000000ULL, 0x3ff71236c0000000ULL,
    0x3ff71208c0000000ULL, 0x3ff711da80000000ULL, 0x3ff711ac80000000ULL,
    0x3ff7117e80000000ULL, 0x3ff7115040000000ULL, 0x3ff7112240000000ULL,
    0x3ff710f400000000ULL, 0x3ff710c600000000ULL, 0x3ff7109800000000ULL,
    0x3ff71069c0000000ULL, 0x3ff7103bc0000000ULL, 0x3ff7100d80000000ULL,
    0x3ff70fdf80000000ULL, 0x3ff70fb180000000ULL, 0x3ff70f8340000000ULL,
    0x3ff70f5540000000ULL, 0x3ff70f2740000000ULL, 0x3ff70ef940000000ULL,
    0x3ff70ecb00000000ULL, 0x3ff70e9d00000000ULL, 0x3ff70e6f00000000ULL,
    0x3ff70e4100000000ULL, 0x3ff70e12c0000000ULL, 0x3ff70de4c0000000ULL,
    0x3ff70db6c0000000ULL, 0x3ff70d88c0000000ULL, 0x3ff70d5ac0000000ULL,
    0x3ff70d2c80000000ULL, 0x3ff70cfe80000000ULL, 0x3ff70cd080000000ULL,
    0x3ff70ca280000000ULL, 0x3ff70c7480000000ULL, 0x3ff70c4680000000ULL,
    0x3ff70c1880000000ULL, 0x3ff70bea80000000ULL, 0x3ff70bbc80000000ULL,
    0x3ff70b8e40000000ULL, 0x3ff70b6040000000ULL, 0x3ff70b3240000000ULL,
    0x3ff70b0440000000ULL, 0x3ff70ad640000000ULL, 0x3ff70aa840000000ULL,
    0x3ff70a7a40000000ULL, 0x3ff71b5380000000ULL, 0x3ff71b3c40000000ULL,
    0x3ff71b2540000000ULL, 0x3ff71b0e00000000ULL, 0x3ff71af700000000ULL,
    0x3ff71adfc0000000ULL, 0x3ff71ac8c0000000ULL, 0x3ff71ab180000000ULL,
    0x3ff71a9a80000000ULL, 0x3ff71a8340000000ULL, 0x3ff71a6c40000000ULL,
    0x3ff71a5500000000ULL, 0x3ff71a3e00000000ULL, 0x3ff71a26c0000000ULL,
    0x3ff71a0fc0000000ULL, 0x3ff719f880000000ULL, 0x3ff719e180000000ULL,
    0x3ff719ca40000000ULL, 0x3ff719b340000000ULL, 0x3ff7199c00000000ULL,
    0x3ff7198500000000ULL, 0x3ff7196dc0000000ULL, 0x3ff71956c0000000ULL,
    0x3ff7193f80000000ULL, 0x3ff7192880000000ULL, 0x3ff7191140000000ULL,
    0x3ff718fa40000000ULL, 0x3ff718e340000000ULL, 0x3ff718cc00000000ULL,
    0x3ff718b500000000ULL, 0x3ff7189dc0000000ULL, 0x3ff71886c0000000ULL,
    0x3ff7186f80000000ULL, 0x3ff7185880000000ULL, 0x3ff7184180000000ULL,
    0x3ff7182a40000000ULL, 0x3ff7181340000000ULL, 0x3ff717fc00000000ULL,
    0x3ff717e500000000ULL, 0x3ff717cdc0000000ULL, 0x3ff717b6c0000000ULL,
    0x3ff7179fc0000000ULL, 0x3ff7178880000000ULL, 0x3ff7177180000000ULL,
    0x3ff7175a40000000ULL, 0x3ff7174340000000ULL, 0x3ff7172c40000000ULL,
    0x3ff7171500000000ULL, 0x3ff716fe00000000ULL, 0x3ff716e6c0000000ULL,
    0x3ff716cfc0000000ULL, 0x3ff716b8c0000000ULL, 0x3ff716a180000000ULL,
    0x3ff7168a80000000ULL, 0x3ff7167380000000ULL, 0x3ff7165c40000000ULL,
    0x3ff7164540000000ULL, 0x3ff7162e00000000ULL, 0x3ff7161700000000ULL,
    0x3ff7160000000000ULL, 0x3ff715e8c0000000ULL, 0x3ff715d1c0000000ULL,
    0x3ff715bac0000000ULL, 0x3ff715a380000000ULL, 0x3ff7158c80000000ULL,
    0x3ff7157580000000ULL, 0x3ff7155e40000000ULL, 0x3ff7154740000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x3f07201750000000ULL,
    0x3d099925641f4f36ULL, 0x3f17102e3b000000ULL, 0x3d120082c162d124ULL,
    0x3f2150340c800000ULL, 0x3d441ee709125ff4ULL, 0x3f27185c92000000ULL,
    0x3ceecd1c8babf46fULL, 0x3f2cd8905d000000ULL, 0x3d3bee60f0bbf0cfULL,
    0x3f315067ff400000ULL, 0x3d3fdbdf8a9648ddULL, 0x3f34308d63400000ULL,
    0x3d3681cbaee1b670ULL, 0x3f3714b8c2800000ULL, 0x3d4102e6bf3d153eULL,
    0x3f39f8e9ed400000ULL, 0x3d365ef233794abcULL, 0x3f3cd92093400000ULL,
    0x3d4fdf0b3d384508ULL, 0x3f3fbd5d4d400000ULL, 0x3d19a166782c348aULL,
    0x3f414ecfb9400000ULL, 0x3d48aba039dde07fULL, 0x3f42c0f3dde00000ULL,
    0x3d435f8a9d5fd823ULL, 0x3f44311ab0600000ULL, 0x3d47963a12458becULL,
    0x3f45a3449ce00000ULL, 0x3d47e1512c8c13fdULL, 0x3f4713712f600000ULL,
    0x3d48d0becb0bd4faULL, 0x3f4885a0e4000000ULL, 0x3d39db21dce1a474ULL,
    0x3f49f5d336a00000ULL, 0x3d48916655660916ULL, 0x3f4b6808b3800000ULL,
    0x3d3d7958c862a7d0ULL, 0x3f4cd840c6800000ULL, 0x3d4312bad43b70f4ULL,
    0x3f4e487bb7c00000ULL, 0x3d46363f975e2c41ULL, 0x3f4fbab9df600000ULL,
    0x3d224e63850fc6c3ULL, 0x3f50957d48a00000ULL, 0x3d25689617a21aa6ULL,
    0x3f514e9f40c00000ULL, 0x3d32ae5fa4582824ULL, 0x3f5206c27a100000ULL,
    0x3d45c3f884b0fd57ULL, 0x3f52bee722a00000ULL, 0x3d4a525c4aad1649ULL,
    0x3f53780d70800000ULL, 0x3d1deb18275071deULL, 0x3f543034f9900000ULL,
    0x3d48eae4d5d75fb6ULL, 0x3f54e95e2c000000ULL, 0x3d3728c5c1291b85ULL,
    0x3f55a18895b00000ULL, 0x3d4f699aabecf0d7ULL, 0x3f5659b46ed00000ULL,
    0x3d22ed258f2d1fa9ULL, 0x3f5712e1f7400000ULL, 0x3d3445ef8b30e580ULL,
    0x3f57cb10b1100000ULL, 0x3d431d52dc75fac6ULL, 0x3f588340da500000ULL,
    0x3d3ffbcba135bd69ULL, 0x3f593b7273000000ULL, 0x3d3871002d63e5dbULL,
    0x3f59f4a5c3200000ULL, 0x3d45afdd1c0bb062ULL, 0x3f5aacda3cc00000ULL,
    0x3d41914fdcd040aeULL, 0x3f5b651025e00000ULL, 0x3d3e7387db245b1fULL,
    0x3f5c1d477e800000ULL, 0x3d4a1e1b593d6b3fULL, 0x3f5cd68096c00000ULL,
    0x3cea7d5ccc31fc26ULL, 0x3f5d8ebad0800000ULL, 0x3d2befbc762a3069ULL,
    0x3f5e46f679d00000ULL, 0x3d4d39ff952be02cULL, 0x3f5eff3392d00000ULL,
    0x3d3d34373b1a1cb4ULL, 0x3f5fb7721b700000ULL, 0x3d2a49f257218470ULL,
    0x3f60385936d80000ULL, 0x3d43adb6602ba3b5ULL, 0x3f609479ebd00000ULL,
    0x3d4a60560591ee13ULL, 0x3f60f09b58a80000ULL, 0x3d343eec4faff44bULL,
    0x3f614cbd7d580000ULL, 0x3d3cc4c6130df139ULL, 0x3f61a8e059e80000ULL,
    0x3d42a0ad9380107fULL, 0x3f620503ee600000ULL, 0x3d12915ab9035a2aULL,
    0x3f6261283ab80000ULL, 0x3d31dded20caaca3ULL, 0x3f62bd4d3ef80000ULL,
    0x3d41ef6e281079c7ULL, 0x3f631972fb280000ULL, 0x3d31986b6b98497fULL,
    0x3f637619a5400000ULL, 0x3d4c369791f2b430ULL, 0x3f63d240d2500000ULL,
    0x3d40eca7ce1c0762ULL, 0x3f642e68b7500000ULL, 0x3d45fa86073b1e2eULL,
    0x3f648a9154480000ULL, 0x3d46d9e905a622fdULL, 0x3f64e6baa9380000ULL,
    0x3d4f04cc4b55a072ULL, 0x3f6542e4b6280000ULL, 0x3d49f57016094e0eULL,
    0x3f659f0f7b180000ULL, 0x3d432659616cf239ULL, 0xbf582dd569800000ULL,
    0xbd3ab5413ed708f7ULL, 0xbf57d0ed91900000ULL, 0xbd4135c67b358e46ULL,
    0xbf5775051b300000ULL, 0xbcfef3c0f6ed6fdeULL, 0xbf57181c89400000ULL,
    0xbd46a4555af7807fULL, 0xbf56bc335ae00000ULL, 0xbd452637919c892dULL,
    0xbf565f4a0f000000ULL, 0xbd46d380806e8ed0ULL, 0xbf56036028a00000ULL,
    0xbd4e545efedc7d6bULL, 0xbf55a67622d00000ULL, 0xbd28c45afdab27bcULL,
    0xbf554a8b84700000ULL, 0xbd40efc094f1d540ULL, 0xbf54eda0c4900000ULL,
    0xbd492aa16e5d7e55ULL, 0xbf5491b56e400000ULL, 0xbd065c31ae3a4995ULL,
    0xbf5434c9f4600000ULL, 0xbd20b59190b30c51ULL, 0xbf53d8dde6000000ULL,
    0xbd208e1f5c64292cULL, 0xbf537bf1b2100000ULL, 0xbd4ba65df606f2e6ULL,
    0xbf532004ebb00000ULL, 0xbd3b2bd51a593ca0ULL, 0xbf52c317fdc00000ULL,
    0xbd440246f5360f7dULL, 0xbf52672a7f500000ULL, 0xbd4229817bbfca47ULL,
    0xbf520a3cd7600000ULL, 0xbd0ad92acf30f123ULL, 0xbf51ae4ea0e00000ULL,
    0xbd292a75693830adULL, 0xbf5151603ed00000ULL, 0xbd49146ee0391426ULL,
    0xbf50f57150400000ULL, 0xbd4e654957002344ULL, 0xbf50988234300000ULL,
    0xbd3d45878f311f09ULL, 0xbf503c928d900000ULL, 0xbd3dca9628fe4ea6ULL,
    0xbf4fbf456ec00000ULL, 0xbd46c4340f0a4c7fULL, 0xbf4f0764b1600000ULL,
    0xbd386c454f8f13f5ULL, 0xbf4e4d8390e00000ULL, 0xbd2790734f099d6aULL,
    0xbf4d95a163400000ULL, 0xbd3586152bf804acULL, 0xbf4cddbe7e800000ULL,
    0xbd43f2dceda75e42ULL, 0xbf4c23db30a00000ULL, 0xbd4ef6f7d23862d0ULL,
    0xbf4b6bf6dbc00000ULL, 0xbd3de7d12300f78cULL, 0xbf4ab21219a00000ULL,
    0xbd4d3c99bb645928ULL, 0xbf49fa2c54800000ULL, 0xbd3c87d4adf18185ULL,
    0xbf4940461e200000ULL, 0xbd49fe5a1a0618b0ULL, 0xbf48885ee8c00000ULL,
    0xbd2934a5aaeef6a6ULL, 0xbf47d076fc200000ULL, 0xbd39cfc0ecbb5462ULL,
    0xbf47168e98600000ULL, 0xbd36f34ebfa1c16bULL, 0xbf465ea53b800000ULL,
    0xbd22ed1c9088ca01ULL, 0xbf45a4bb63600000ULL, 0xbd4233202d6de9b5ULL,
    0xbf44ecd096200000ULL, 0xbd48587110d84808ULL, 0xbf4432e549c00000ULL,
    0xbd3dc6145992900aULL, 0xbf437af90c200000ULL, 0xbd4c28b1f611f4f2ULL,
    0xbf42c30c17600000ULL, 0xbd49b47212496da4ULL, 0xbf42091e9d800000ULL,
    0xbd22545fb07b63e5ULL, 0xbf41513038600000ULL, 0xbcf3fc746b4daa19ULL,
    0xbf4097414a000000ULL, 0xbd2819ad9f7943acULL, 0xbf3fbea2e8c00000ULL,
    0xbd4acc9205018f01ULL, 0xbf3e4ec1cf400000ULL, 0xbd48b0285decad9cULL,
    0xbf3cdadf97400000ULL, 0xbd4ca2dd63115207ULL, 0xbf3b6afb9d000000ULL,
    0xbd354003a7fc9363ULL, 0xbf39f7167c000000ULL, 0xbd4a8ae0407a7831ULL,
    0xbf38872fa0c00000ULL, 0xbd42dd8b91a8939dULL, 0xbf37174757000000ULL,
    0xbd3a49252f243d55ULL, 0xbf35a35ddac00000ULL, 0xbd2741df22a2581bULL,
    0xbf343372b0000000ULL, 0xbd0cab753c22e0d2ULL, 0xbf32c38616800000ULL,
    0xbd4b6e115fc1d4ddULL, 0xbf314f983ec00000ULL, 0xbd3f929a97121e28ULL,
    0xbf2fbf5189000000ULL, 0xbd13ea1e806fdbdaULL, 0xbf2cd77007000000ULL,
    0xbd370eb151e95ecbULL, 0xbf29f78b50000000ULL, 0xbd418759ff247507ULL,
    0xbf2717a3bc000000ULL, 0xbd3c0d6bc6c6f140ULL, 0xbf242fb982800000ULL,
    0xbd4cf2efb8d3c162ULL, 0xbf214fcc2c800000ULL, 0xbd206ce312cfe97eULL,
    0xbf1cdfb7f1000000ULL, 0xbd4d77341056af68ULL, 0xbf170fd210000000ULL,
    0xbd24caa429329aefULL, 0xbf114fe624000000ULL, 0xbcf284c306b71311ULL,
    0xbf071fe8fa000000ULL, 0xbd33bb0cdd98569aULL, 0xbef6fff4ac000000ULL,
    0xbd4d54d70b7407c7ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x3ff0000000000000ULL, 0x0000000000000000ULL, 0x3ff0163da9fb3335ULL,
    0x3c9b61299ab8cdb7ULL, 0x3ff02c9a3e778061ULL, 0xbc719083535b085dULL,
    0x3ff04315e86e7f85ULL, 0xbc90a31c1977c96eULL, 0x3ff059b0d3158574ULL,
    0x3c8d73e2a475b465ULL, 0x3ff0706b29ddf6deULL, 0xbc8c91dfe2b13c26ULL,
    0x3ff0874518759bc8ULL, 0x3c6186be4bb284ffULL, 0x3ff09e3ecac6f383ULL,
    0x3c91487818316135ULL, 0x3ff0b5586cf9890fULL, 0x3c98a62e4adc610aULL,
    0x3ff0cc922b7247f7ULL, 0x3c901edc16e24f71ULL, 0x3ff0e3ec32d3d1a2ULL,
    0x3c403a1727c57b52ULL, 0x3ff0fb66affed31bULL, 0xbc6b9bedc44ebd7bULL,
    0x3ff11301d0125b51ULL, 0xbc96c51039449b39ULL, 0x3ff12abdc06c31ccULL,
    0xbc51b514b36ca5c7ULL, 0x3ff1429aaea92de0ULL, 0xbc932fbf9af1369eULL,
    0x3ff15a98c8a58e51ULL, 0x3c82406ab9eeab09ULL, 0x3ff172b83c7d517bULL,
    0xbc819041b9d78a75ULL, 0x3ff18af9388c8deaULL, 0xbc911023d1970f6bULL,
    0x3ff1a35beb6fcb75ULL, 0x3c8e5b4c7b4968e4ULL, 0x3ff1bbe084045cd4ULL,
    0xbc995386352ef607ULL, 0x3ff1d4873168b9aaULL, 0x3c9e016e00a2643cULL,
    0x3ff1ed5022fcd91dULL, 0xbc91df98027bb78bULL, 0x3ff2063b88628cd6ULL,
    0x3c8dc775814a8494ULL, 0x3ff21f49917ddc96ULL, 0x3c82a97e9494a5edULL,
    0x3ff2387a6e756238ULL, 0x3c99b07eb6c70572ULL, 0x3ff251ce4fb2a63fULL,
    0x3c8ac155bef4f4a4ULL, 0x3ff26b4565e27cddULL, 0x3c82bd339940e9d9ULL,
    0x3ff284dfe1f56381ULL, 0xbc9a4c3a8c3f0d7dULL, 0x3ff29e9df51fdee1ULL,
    0x3c8612e8afad1255ULL, 0x3ff2b87fd0dad990ULL, 0xbc410adcd6381aa3ULL,
    0x3ff2d285a6e4030bULL, 0x3c90024754db41d4ULL, 0x3ff2ecafa93e2f56ULL,
    0x3c71ca0f45d52383ULL, 0x3ff306fe0a31b715ULL, 0x3c86f46ad23182e4ULL,
    0x3ff32170fc4cd831ULL, 0x3c8a9ce78e18047cULL, 0x3ff33c08b26416ffULL,
    0x3c932721843659a5ULL, 0x3ff356c55f929ff1ULL, 0xbc8b5cee5c4e4628ULL,
    0x3ff371a7373aa9cbULL, 0xbc963aeabf42eae1ULL, 0x3ff38cae6d05d866ULL,
    0xbc9e958d3c9904bcULL, 0x3ff3a7db34e59ff7ULL, 0xbc75e436d661f5e2ULL,
    0x3ff3c32dc313a8e5ULL, 0xbc9efff8375d29c3ULL, 0x3ff3dea64c123422ULL,
    0x3c8ada0911f09ebbULL, 0x3ff3fa4504ac801cULL, 0xbc97d023f956f9f3ULL,
    0x3ff4160a21f72e2aULL, 0xbc5ef3691c309278ULL, 0x3ff431f5d950a897ULL,
    0xbc81c7dde35f7998ULL, 0x3ff44e086061892dULL, 0x3c489b7a04ef80cfULL,
    0x3ff46a41ed1d0057ULL, 0x3c9c944bd1648a76ULL, 0x3ff486a2b5c13cd0ULL,
    0x3c73c1a3b69062f0ULL, 0x3ff4a32af0d7d3deULL, 0x3c99cb62f3d1be56ULL,
    0x3ff4bfdad5362a27ULL, 0x3c7d4397afec42e2ULL, 0x3ff4dcb299fddd0dULL,
    0x3c98ecdbbc6a7833ULL, 0x3ff4f9b2769d2ca7ULL, 0xbc94b309d25957e3ULL,
    0x3ff516daa2cf6642ULL, 0xbc8f768569bd93eeULL, 0x3ff5342b569d4f82ULL,
    0xbc807abe1db13cacULL, 0x3ff551a4ca5d920fULL, 0xbc8d689cefede59aULL,
    0x3ff56f4736b527daULL, 0x3c99bb2c011d93acULL, 0x3ff58d12d497c7fdULL,
    0x3c8295e15b9a1de7ULL, 0x3ff5ab07dd485429ULL, 0x3c96324c054647acULL,
    0x3ff5c9268a5946b7ULL, 0x3c3c4b1b816986a2ULL, 0x3ff5e76f15ad2148ULL,
    0x3c9ba6f93080e65dULL, 0x3ff605e1b976dc09ULL, 0xbc93e2429b56de47ULL,
    0x3ff6247eb03a5585ULL, 0xbc9383c17e40b496ULL, 0x3ff6434634ccc320ULL,
    0xbc8c483c759d8932ULL, 0x3ff6623882552225ULL, 0xbc9bb60987591c33ULL,
    0x3ff68155d44ca973ULL, 0x3c6038ae44f73e64ULL, 0x3ff6a09e667f3bcdULL,
    0xbc9bdd3413b26455ULL, 0x3ff6c012750bdabfULL, 0xbc72895667ff0b0cULL,
    0x3ff6dfb23c651a2fULL, 0xbc6bbe3a683c88aaULL, 0x3ff6ff7df9519484ULL,
    0xbc883c0f25860ef6ULL, 0x3ff71f75e8ec5f74ULL, 0xbc816e4786887a99ULL,
    0x3ff73f9a48a58174ULL, 0xbc90a8d96c65d53bULL, 0x3ff75feb564267c9ULL,
    0xbc90245957316dd3ULL, 0x3ff780694fde5d3fULL, 0x3c9866b80a02162cULL,
    0x3ff7a11473eb0187ULL, 0xbc841577ee04992fULL, 0x3ff7c1ed0130c132ULL,
    0x3c9f124cd1164dd5ULL, 0x3ff7e2f336cf4e62ULL, 0x3c705d02ba15797eULL,
    0x3ff80427543e1a12ULL, 0xbc927c86626d972aULL, 0x3ff82589994cce13ULL,
    0xbc9d4c1dd41532d7ULL, 0x3ff8471a4623c7adULL, 0xbc88d684a341cdfbULL,
    0x3ff868d99b4492edULL, 0xbc9fc6f89bd4f6baULL, 0x3ff88ac7d98a6699ULL,
    0x3c9994c2f37cb53aULL, 0x3ff8ace5422aa0dbULL, 0x3c96e9f156864b26ULL,
    0x3ff8cf3216b5448cULL, 0xbc70d55e32e9e3aaULL, 0x3ff8f1ae99157736ULL,
    0x3c85cc13a2e3976cULL, 0x3ff9145b0b91ffc6ULL, 0xbc9dd6792e582523ULL,
    0x3ff93737b0cdc5e5ULL, 0xbc675fc781b57ebbULL, 0x3ff95a44cbc8520fULL,
    0xbc764b7c96a5f039ULL, 0x3ff97d829fde4e50ULL, 0xbc9d185b7c1b85d0ULL,
    0x3ff9a0f170ca07baULL, 0xbc9173bd91cee632ULL, 0x3ff9c49182a3f090ULL,
    0x3c7c7c46b071f2beULL, 0x3ff9e86319e32323ULL, 0x3c7824ca78e64c6eULL,
    0x3ffa0c667b5de565ULL, 0xbc9359495d1cd532ULL, 0x3ffa309bec4a2d33ULL,
    0x3c96305c7ddc36abULL, 0x3ffa5503b23e255dULL, 0xbc9d2f6edb8d41e1ULL,
    0x3ffa799e1330b358ULL, 0x3c9bcb7ecac563c6ULL, 0x3ffa9e6b5579fdbfULL,
    0x3c90fac90ef7fd31ULL, 0x3ffac36bbfd3f37aULL, 0xbc8f9234cae76cd0ULL,
    0x3ffae89f995ad3adULL, 0x3c97a1cd345dcc81ULL, 0x3ffb0e07298db666ULL,
    0xbc9bdef54c80e424ULL, 0x3ffb33a2b84f15fbULL, 0xbc62805e3084d707ULL,
    0x3ffb59728de5593aULL, 0xbc9c71dfbbba6de3ULL, 0x3ffb7f76f2fb5e47ULL,
    0xbc75584f7e54ac3aULL, 0x3ffba5b030a1064aULL, 0xbc9efcd30e54292eULL,
    0x3ffbcc1e904bc1d2ULL, 0x3c823dd07a2d9e84ULL, 0x3ffbf2c25bd71e09ULL,
    0xbc9efdca3f6b9c72ULL, 0x3ffc199bdd85529cULL, 0x3c811065895048ddULL,
    0x3ffc40ab5fffd07aULL, 0x3c9b4537e083c60aULL, 0x3ffc67f12e57d14bULL,
    0x3c92884dff483cacULL, 0x3ffc8f6d9406e7b5ULL, 0x3c71acbc48805c44ULL,
    0x3ffcb720dcef9069ULL, 0x3c7503cbd1e949dbULL, 0x3ffcdf0b555dc3faULL,
    0xbc8dd83b53829d72ULL, 0x3ffd072d4a07897cULL, 0xbc9cbc3743797a9cULL,
    0x3ffd2f87080d89f2ULL, 0xbc9d487b719d8577ULL, 0x3ffd5818dcfba487ULL,
    0x3c82ed02d75b3706ULL, 0x3ffd80e316c98398ULL, 0xbc911ec18beddfe8ULL,
    0x3ffda9e603db3285ULL, 0x3c9c2300696db532ULL, 0x3ffdd321f301b460ULL,
    0x3c92da5778f018c2ULL, 0x3ffdfc97337b9b5fULL, 0xbc91a5cd4f184b5bULL,
    0x3ffe264614f5a129ULL, 0xbc97b627817a1496ULL, 0x3ffe502ee78b3ff6ULL,
    0x3c839e8980a9cc8fULL, 0x3ffe7a51fbc74c83ULL, 0x3c92d522ca0c8de1ULL,
    0x3ffea4afa2a490daULL, 0xbc9e9c23179c2893ULL, 0x3ffecf482d8e67f1ULL,
    0xbc9c93f3b411ad8cULL, 0x3ffefa1bee615a27ULL, 0x3c9dc7f486a4b6b0ULL,
    0x3fff252b376bba97ULL, 0x3c93a1a5bf0d8e43ULL, 0x3fff50765b6e4540ULL,
    0x3c99d3e12dd8a18aULL, 0x3fff7bfdad9cbe14ULL, 0xbc9dbb12d0063509ULL,
    0x3fffa7c1819e90d8ULL, 0x3c874853f3a5931eULL, 0x3fffd3c22b8f71f1ULL,
    0x3c62eb74966579e7ULL, 0x3e79c3a6966457e8ULL, 0xbfd62e4346694107ULL,
    0x3fc47fd462b6dee1ULL, 0xbfb550472a9012d8ULL, 0x3fe62e42fefa39efULL,
    0x3fcebfbdff82c58fULL, 0x3fac6b08d704a0c0ULL, 0x3f83b2ab6fba4e77ULL,
    0x3f55d87fe78a6731ULL, 0x7fe0000000000000ULL, 0x0010000000000000ULL,
    0x0000000000000000ULL, 0x3ff0000000000000ULL, 0xbff0000000000000ULL,
    0x42c8000000000000ULL, 0x3ff7154740000000ULL, 0x41a0000002000000ULL,
    0x4c70000000000000ULL, 0x3370000000000000ULL
};

/* 32*DblRcp at the start of each 1024-prefix bucket */
static const unsigned char pow_rcp_base[64] = {
    64, 63, 62, 61, 60, 59, 59, 58, 57, 56, 55, 55, 54, 53, 53, 52,
    51, 51, 50, 49, 49, 48, 48, 47, 47, 46, 46, 45, 45, 44, 44, 43,
    43, 42, 42, 41, 41, 41, 40, 40, 39, 39, 39, 38, 38, 38, 37, 37,
    37, 36, 36, 36, 35, 35, 35, 34, 34, 34, 34, 33, 33, 33, 33, 32
};
/* offset of the single step-down point inside each bucket (1024 = none) */
static const unsigned short pow_rcp_boff[64] = {
    518, 551, 618, 720, 861, 1024, 15, 242, 509, 822, 1024, 159, 576, 1024, 18, 549,
    1024, 110, 766, 1024, 466, 1024, 239, 1024, 87, 1024, 23, 1024, 45, 1024, 165, 1024,
    386, 1024, 717, 1024, 1024, 138, 1024, 715, 1024, 1024, 399, 1024, 1024, 233, 1024, 1024,
    224, 1024, 1024, 392, 1024, 1024, 742, 1024, 1024, 1024, 277, 1024, 1024, 1024, 35, 1024
};

#define DBL_MAX_BITS     0x7fefffffffffffffULL
#define POS_INF_BITS     0x7ff0000000000000ULL
#define NEG_INF_BITS     0xfff0000000000000ULL
#define QUIET_BIT        0x0008000000000000ULL
/* SSE result of an invalid operation without NaN operands (e.g. 0.0/0.0) */
#define X86_DEFAULT_NAN  0xfff8000000000000ULL

/* ---------------------------------------------------------------------------
 * Exact directed rounding in round-to-nearest mode.
 *
 * soft_fma_dir() evaluates a*b + c exactly with integer arithmetic and
 * rounds the exact value toward zero or toward -inf.  It is valid for all
 * finite a, b, c and is the reference for the fast paths; the fast paths
 * only handle operand ranges where the error-free transformations used are
 * proven exact and otherwise defer to it.
 * ------------------------------------------------------------------------- */

#define DIR_RZ 0
#define DIR_RD 1

typedef struct {
    uint64_t w[4];              /* 256-bit magnitude, w[0] least significant */
} wide_t;

static void wide_zero(wide_t *a)
{
    a->w[0] = a->w[1] = a->w[2] = a->w[3] = 0;
}

static int wide_is_zero(const wide_t *a)
{
    return (a->w[0] | a->w[1] | a->w[2] | a->w[3]) == 0;
}

static int bitlen64(uint64_t v)
{
    int n = 0;
    while (v) {
        v >>= 1;
        n++;
    }
    return n;
}

/* index of the most significant set bit, -1 for zero */
static int wide_top(const wide_t *a)
{
    int i;
    for (i = 3; i >= 0; i--)
        if (a->w[i])
            return 64 * i + bitlen64(a->w[i]) - 1;
    return -1;
}

/* a <<= n for 0 <= n < 256; callers guarantee no set bit is shifted out */
static void wide_shl(wide_t *a, int n)
{
    int q = n >> 6, r = n & 63, i;
    for (i = 3; i >= 0; i--) {
        uint64_t v = 0;
        if (i - q >= 0) {
            v = a->w[i - q] << r;
            if (r && i - q - 1 >= 0)
                v |= a->w[i - q - 1] >> (64 - r);
        }
        a->w[i] = v;
    }
}

/* a >>= n for n >= 0; returns nonzero if a set bit was shifted out */
static int wide_shr(wide_t *a, int n)
{
    int lost = 0, q, r, i;
    if (n >= 256) {
        lost = !wide_is_zero(a);
        wide_zero(a);
        return lost;
    }
    q = n >> 6;
    r = n & 63;
    for (i = 0; i < q; i++)
        lost |= (a->w[i] != 0);
    if (r)
        lost |= ((a->w[q] & ((1ULL << r) - 1)) != 0);
    for (i = 0; i < 4; i++) {
        uint64_t v = 0;
        if (i + q < 4) {
            v = a->w[i + q] >> r;
            if (r && i + q + 1 < 4)
                v |= a->w[i + q + 1] << (64 - r);
        }
        a->w[i] = v;
    }
    return lost;
}

static int wide_cmp(const wide_t *a, const wide_t *b)
{
    int i;
    for (i = 3; i >= 0; i--)
        if (a->w[i] != b->w[i])
            return a->w[i] > b->w[i] ? 1 : -1;
    return 0;
}

static void wide_add(wide_t *r, const wide_t *a, const wide_t *b)
{
    uint64_t carry = 0;
    int i;
    for (i = 0; i < 4; i++) {
        uint64_t s = a->w[i] + b->w[i];
        uint64_t c1 = s < a->w[i];
        uint64_t t = s + carry;
        uint64_t c2 = t < s;
        r->w[i] = t;
        carry = c1 | c2;
    }
}

/* r = a - b, requires a >= b */
static void wide_sub(wide_t *r, const wide_t *a, const wide_t *b)
{
    uint64_t borrow = 0;
    int i;
    for (i = 0; i < 4; i++) {
        uint64_t d = a->w[i] - b->w[i];
        uint64_t b1 = a->w[i] < b->w[i];
        uint64_t t = d - borrow;
        uint64_t b2 = d < borrow;
        r->w[i] = t;
        borrow = b1 | b2;
    }
}

static void mul_64x64(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo)
{
    uint64_t a0 = a & 0xffffffffULL, a1 = a >> 32;
    uint64_t b0 = b & 0xffffffffULL, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    *lo = (p00 & 0xffffffffULL) | (mid << 32);
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

/* finite v = (-1)^s * m * 2^e with integer m < 2^53 */
static void split_double(double v, int *s, uint64_t *m, int *e)
{
    uint64_t u = as_u64(v);
    int be = (int)((u >> 52) & 0x7ff);
    *s = (int)(u >> 63);
    *m = u & 0x000fffffffffffffULL;
    if (be == 0) {
        *e = -1074;
    } else {
        *m |= 0x0010000000000000ULL;
        *e = be - 1075;
    }
}

/* shift x so that its value is x * 2^base; set a sticky bit if bits are lost */
static void wide_align(wide_t *x, int e, int base)
{
    int d = e - base;
    if (d >= 0) {
        wide_shl(x, d);
    } else if (wide_shr(x, -d)) {
        x->w[0] |= 1;
    }
}

/*
 * Exact a*b + c rounded toward zero (DIR_RZ) or toward -inf (DIR_RD) for
 * finite a, b, c.  The larger operand is placed with its leading bit at
 * position 200 of a 256-bit integer.  The smaller one is either exact or
 * shifted right with a sticky bit in position 0, and in the sticky case
 * the result keeps at least 147 bits above position 0.  The truncated sum
 * is then an odd integer within one unit of the exact sum, so it rounds
 * exactly like the exact sum on any grid of spacing >= 2 units.
 */
static double soft_fma_dir(double a, double b, double c, int dir)
{
    int sa, sb, sc, ea, eb, ec, sp, sr, top, base, h, q, f, inexact;
    uint64_t ma, mb, mc, phi, plo, k;
    wide_t P, C, R;
    double v;

    split_double(a, &sa, &ma, &ea);
    split_double(b, &sb, &mb, &eb);
    split_double(c, &sc, &mc, &ec);
    sp = sa ^ sb;
    if (ma == 0 || mb == 0) {
        if (mc != 0 || sp == sc)
            return c;
        return dir == DIR_RD ? -0.0 : 0.0;
    }
    mul_64x64(ma, mb, &phi, &plo);
    wide_zero(&P);
    P.w[0] = plo;
    P.w[1] = phi;
    top = ea + eb + wide_top(&P);
    wide_zero(&C);
    if (mc != 0) {
        int tc;
        C.w[0] = mc;
        tc = ec + wide_top(&C);
        if (tc > top)
            top = tc;
    }
    base = top - 200;
    wide_align(&P, ea + eb, base);
    if (mc != 0)
        wide_align(&C, ec, base);

    if (mc == 0 || sp == sc) {
        wide_add(&R, &P, &C);
        sr = sp;
    } else {
        int cmp = wide_cmp(&P, &C);
        if (cmp == 0)
            return dir == DIR_RD ? -0.0 : 0.0;
        if (cmp > 0) {
            wide_sub(&R, &P, &C);
            sr = sp;
        } else {
            wide_sub(&R, &C, &P);
            sr = sc;
        }
    }

    h = wide_top(&R);
    if (h + base >= -1022)
        q = h - 52;
    else
        q = -1074 - base;
    if (q <= 0) {
        k = R.w[0];
        f = base;
        inexact = 0;
    } else {
        inexact = wide_shr(&R, q);
        k = R.w[0];
        f = base + q;
    }
    if (dir == DIR_RD && sr && inexact)
        k += 1;
    if (k == 0)
        return sr ? -0.0 : 0.0;
    if (f + bitlen64(k) - 1 > 1023) {
        if (dir == DIR_RD && sr)
            return as_f64(NEG_INF_BITS);
        return sr ? -as_f64(DBL_MAX_BITS) : as_f64(DBL_MAX_BITS);
    }
    v = ldexp((double)k, f);
    return sr ? -v : v;
}

/* Knuth's TwoSum: s + t == a + b exactly when no overflow occurs */
static void two_sum(double a, double b, double *s, double *t)
{
    double ss = a + b;
    double bb = ss - a;
    double aa = ss - bb;
    *s = ss;
    *t = (a - aa) + (b - bb);
}

/*
 * Exact sign of a0 + a1 + a2 + a3 (no overflow).  Shewchuk's
 * Grow-Expansion turns the terms into a nonoverlapping expansion whose
 * nonzero components increase in magnitude; the sign of the sum is the
 * sign of the last nonzero component.
 */
static int expansion_sign4(double a0, double a1, double a2, double a3)
{
    double e[4], in[3], q, h;
    int n = 1, i, j;
    e[0] = a0;
    in[0] = a1;
    in[1] = a2;
    in[2] = a3;
    for (j = 0; j < 3; j++) {
        q = in[j];
        for (i = 0; i < n; i++) {
            two_sum(q, e[i], &q, &h);
            e[i] = h;
        }
        e[n++] = q;
    }
    for (i = n - 1; i >= 0; i--)
        if (e[i] != 0.0)
            return e[i] > 0.0 ? 1 : -1;
    return 0;
}

/* neighbour of finite nonzero v toward zero */
static double toward_zero(double v)
{
    return as_f64(as_u64(v) - 1);
}

/* neighbour of finite nonzero v toward -inf */
static double toward_neg(double v)
{
    uint64_t u = as_u64(v);
    return as_f64(v > 0.0 ? u - 1 : u + 1);
}

/*
 * Fast-path bounds.  If 2^-968 <= |RN(a*b)| then e(a) + e(b) >= -970, so
 * a*b - RN(a*b) is representable and fma(a, b, -RN(a*b)) is exact.  The
 * upper bound keeps every TwoSum below 2^1022 (no overflow).
 */
#define EFT_LO 0x1p-968
#define EFT_HI 0x1p1020

/* vmulpd {rz-sae} */
static double mul_rz(double a, double b)
{
    double p = a * b;
    double ap = fabs(p);
    if (ap >= EFT_LO && ap <= EFT_HI) {
        double e = fma(a, b, -p);
        if (e != 0.0 && ((p > 0.0) != (e > 0.0)))
            p = toward_zero(p);
        return p;
    }
    return soft_fma_dir(a, b, -0.0, DIR_RZ);
}

/* vfmadd/vfmsub {rz-sae}: a*b + c rounded toward zero */
static double fma_rz(double a, double b, double c)
{
    double r = fma(a, b, c);
    double ph;
    /* a zero RN result is also the RZ result (same sign rules, |S| <= 2^-1075) */
    if (r == 0.0)
        return r;
    /* a*b is an exact zero, so S == c is representable */
    if (a == 0.0 || b == 0.0)
        return r;
    ph = a * b;
    if (fabs(ph) >= EFT_LO && fabs(ph) <= EFT_HI && fabs(c) <= EFT_HI &&
        fabs(r) <= EFT_HI) {
        double pl = fma(a, b, -ph);
        int s = expansion_sign4(ph, pl, c, -r);
        if (s != 0 && ((r > 0.0) == (s < 0)))
            r = toward_zero(r);
        return r;
    }
    return soft_fma_dir(a, b, c, DIR_RZ);
}

/* vaddpd {rz-sae} */
static double add_rz(double a, double b)
{
    if (fabs(a) <= EFT_HI && fabs(b) <= EFT_HI) {
        double s, t;
        two_sum(a, b, &s, &t);
        if (t != 0.0 && ((s > 0.0) != (t > 0.0)))
            s = toward_zero(s);
        return s;
    }
    return soft_fma_dir(a, 1.0, b, DIR_RZ);
}

/* vaddpd {rd-sae} */
static double add_rd(double a, double b)
{
    if (fabs(a) <= EFT_HI && fabs(b) <= EFT_HI) {
        double s, t;
        two_sum(a, b, &s, &t);
        if (t < 0.0)
            return toward_neg(s);
        if (s == 0.0)
            return (as_u64(a) == 0 && as_u64(b) == 0) ? 0.0 : -0.0;
        return s;
    }
    return soft_fma_dir(a, 1.0, b, DIR_RD);
}

/*
 * vreducepd $65 (M = 4, round down): v - floor(16 v)/16, where the
 * subtraction is also performed rounding toward -inf.  For |v| >= 2^48,
 * 16 v is an integer and the result is v - v = -0 (this also keeps 16 v
 * from overflowing).
 */
static double reduce_m4_rd(double v)
{
    double t;
    if (fabs(v) >= 0x1p48)
        return -0.0;
    t = floor(v * 16.0) * 0.0625;
    return add_rd(v, -t);
}

/*
 * vrcp14pd followed by vrndscalepd $88 (round to nearest, 5 fraction bits)
 * for a mantissa m in [1/2, 1).  vrcp14pd only looks at the 16 leading
 * fraction bits p of m (apart from m == 1/2 exactly, which still rounds to
 * 2.0), and 32 * DblRcp is a non-increasing step function of p that drops
 * by one at 32 values of p.  pow_rcp_base/pow_rcp_boff encode those steps
 * in 64 buckets of 1024 prefixes (each bucket holds at most one step).
 */
static double rcp14_rndscale(double m)
{
    unsigned p = (unsigned)((as_u64(m) >> 36) & 0xffff);
    int k32 = pow_rcp_base[p >> 10] - ((p & 1023u) >= pow_rcp_boff[p >> 10]);
    return (double)k32 * 0.03125;
}

/* SSE NaN rules for a two-operand arithmetic instruction */
static double quiet_nan(double v)
{
    return as_f64(as_u64(v) | QUIET_BIT);
}

static int is_nan_bits(double v)
{
    return (as_u64(v) & 0x7fffffffffffffffULL) > POS_INF_BITS;
}

static double sse_addsd(double dst, double src)
{
    if (is_nan_bits(dst))
        return quiet_nan(dst);
    if (is_nan_bits(src))
        return quiet_nan(src);
    return dst + src;
}

/* ---------------------------------------------------------------------------
 * __svml_dpow_ha_cout_rare_internal
 * ------------------------------------------------------------------------- */

#define HI16(v) ((uint32_t)(as_u64(v) >> 48))
#define HI32(v) ((uint32_t)(as_u64(v) >> 32))
#define LO32(v) ((uint32_t)as_u64(v))
#define TOPBYTE(v) ((uint32_t)(as_u64(v) >> 56))

/* two's complement reinterpretation and arithmetic shift without
   implementation-defined behaviour */
static int32_t s32(uint32_t v)
{
    return (v & 0x80000000u) ? -(int32_t)(~v) - 1 : (int32_t)v;
}

static uint32_t sar32(uint32_t v, int n)
{
    return (v >> n) | ((v & 0x80000000u) ? ~(0xffffffffu >> n) : 0u);
}

static double set_hi16(double v, uint32_t w)
{
    uint64_t u = as_u64(v);
    u = (u & 0x0000ffffffffffffULL) | ((uint64_t)(w & 0xffff) << 48);
    return as_f64(u);
}

static double pow_rare(double px, double py)
{
    double xmm0 = 0, xmm1 = 0, xmm2 = 0, xmm3 = 0, xmm4 = 0, xmm5 = 0;
    double xmm6 = 0, xmm7 = 0, xmm8 = 0, xmm9 = 0, xmm10 = 0, xmm11 = 0;
    double xmm12 = 0, xmm13 = 0, xmm14 = 0, xmm15 = 0;
    double m16 = 0, m24 = 0, m32 = 0, m40 = 0, m48 = 0, m56 = 0;
    double m64 = 0, m72 = 0, m80 = 0, m88 = 0;
    uint32_t edx = 0, ecx = 0, esi = 0, edi = 0, r9 = 0, r10 = 0, r11 = 0;
    uint32_t r12 = 0, r13 = 0, r15 = 0, ebx = 0, bl = 0, cl = 0;

    xmm1 = CT(853);
    xmm3 = px;
    xmm0 = py;
    xmm3 = xmm3 * xmm1;
    xmm0 = xmm0 * xmm1;
    if (is_nan_bits(xmm3))
        xmm3 = quiet_nan(xmm3);
    if (is_nan_bits(xmm0))
        xmm0 = quiet_nan(xmm0);
    m72 = xmm3;
    m16 = xmm0;
    r13 = HI16(m72);
    edx = HI16(m16);
    r13 &= 32752;
    r12 = TOPBYTE(m72);
    edx &= 32752;
    bl = TOPBYTE(m16);
    r12 &= 128;
    bl &= 0x80;
    r13 >>= 4;
    edx >>= 4;
    r12 >>= 7;
    bl >>= 7;
    if ((HI32(m72) & 1048575) != 0 || LO32(m72) != 0)
        r11 = 0;
    else
        r11 = 1;

    ecx = HI32(m16);
    edi = LO32(m16);
    ecx &= 1048575;
    if (ecx != 0 || edi != 0)
        r10 = 0;
    else
        r10 = 1;

    r9 = edx;
    esi = (edx != 2047);
    r9 <<= 20;
    r9 |= ecx;
    ecx = edi;
    ecx |= r9;
    if (ecx == 0)
        goto L24;

    ecx = edx - 1023;
    if (s32(edx) < 1023)
        goto L23;
    if (esi == 0)
        goto L23;
    if (s32(ecx) > 20)
        goto L16;

    r15 = r9;
    r15 <<= (ecx & 31);
    r15 <<= 12;
    r15 |= edi;
    if (r15 == 0)
        goto L15;
L14:
    r9 = 0;
    goto L21;
L15:
    ecx = edx - 1012;
    r9 <<= (ecx & 31);
    r9 = ~r9;
    r9 >>= 31;
    r9 += 1;
    goto L21;
L16:
    if (s32(ecx) >= 53)
        goto L20;
    ecx = edx - 1012;
    edi <<= (ecx & 31);
    if ((edi & 2147483647u) != 0)
        goto L14;
    edi = ~edi;
    edi >>= 31;
    r9 = edi + 1;
    goto L21;
L20:
    r9 = 2;
L21:
    if (r12 != 0)
        goto L29;
    if (r13 != 1023)
        goto L29;
    goto L57;
L23:
    r9 = 0;
    goto L25;
L24:
    r9 = 2;
L25:
    if (r12 != 0)
        goto L27;
    if (r13 == 1023)
        goto L74;
L27:
    if (edx != 0)
        goto L29;
    if (r10 != 0)
        goto L55;
L29:
    if (r13 == 2047)
        goto L31;
L30:
    cl = 1;
    goto L32;
L31:
    cl = 0;
L32:
    r11 |= cl;
    if (r11 == 0)
        goto L54;
    r10 |= esi;
    if (r10 == 0)
        goto L54;
    xmm2 = CT(852);
    if (xmm3 == xmm2)
        goto L71;
    if (xmm3 == CT(854))
        goto L68;
L36:
    if (cl == 0)
        goto L47;
    if (esi == 0)
        goto L47;
    if (xmm3 > xmm2)
        goto L40;
    if (r9 == 0)
        goto L46;
L40:
    r9 &= r12;
    m72 = xmm3;
    m72 = as_f64(as_u64(m72) & 0x7fffffffffffffffULL);
    xmm0 = CT(853 + r9);
    ecx = 0;
    if (r13 != 0)
        goto L42;
    xmm2 = m72;
    ecx = (uint32_t)-200;
    xmm2 = xmm2 * CT(858);
    m72 = xmm2;
    goto L43;
L42:
    xmm2 = m72;
L43:
    esi = HI16(m72);
    xmm7 = 0.0;
    esi &= 32752;
    esi >>= 4;
    r9 = HI32(m72);
    esi <<= 20;
    r9 &= 1048575;
    m56 = xmm2;
    esi |= r9;
    edi = HI16(m56);
    esi += (uint32_t)-1072152576;
    edi &= (uint32_t)-32753;
    edi += 16368;
    m56 = set_hi16(m56, edi);
    esi = sar32(esi, 20);
    r10 = HI32(m56);
    esi += ecx;
    r10 &= 1032192;
    r10 += 16384;
    r10 >>= 15;
    xmm2 = m56;
    xmm5 = CT(r10);
    r10 += r10;
    xmm6 = xmm5;
    xmm14 = CT(857);
    xmm6 = xmm6 * xmm2;
    xmm7 = (double)s32(esi);
    xmm14 = xmm14 * xmm2;
    xmm7 = xmm7 + CT(33 + r10);
    m48 = xmm6;
    m32 = xmm14;
    r11 = HI32(m48);
    r11 &= 64512;
    xmm15 = m32;
    r11 += 1024;
    r11 >>= 11;
    xmm15 = xmm15 - m56;
    xmm12 = CT(99 + r11);
    r11 += r11;
    xmm6 = xmm6 * xmm12;
    xmm7 = xmm7 + CT(132 + r11);
    xmm5 = xmm5 * xmm12;
    m24 = xmm15;
    xmm4 = m32;
    xmm3 = m24;
    m40 = xmm6;
    xmm4 = xmm4 - xmm3;
    r12 = HI32(m40);
    r12 &= 4080;
    r12 += 16;
    m32 = xmm4;
    r12 >>= 5;
    xmm12 = m32;
    xmm13 = CT(198 + r12);
    r12 += r12;
    xmm5 = xmm5 * xmm13;
    xmm2 = xmm2 - xmm12;
    xmm7 = xmm7 + CT(327 + r12);
    xmm6 = xmm6 * xmm13;
    m24 = xmm2;
    xmm8 = xmm6;
    xmm2 = CT(857);
    xmm2 = xmm2 * xmm5;
    xmm8 = xmm8 - CT(856);
    xmm3 = m32;
    xmm4 = m24;
    m32 = xmm2;
    xmm13 = m32;
    xmm11 = CT(34 + r10);
    xmm13 = xmm13 - xmm5;
    m24 = xmm13;
    xmm2 = m32;
    xmm14 = m24;
    xmm10 = CT(133 + r11);
    xmm2 = xmm2 - xmm14;
    m32 = xmm2;
    xmm2 = xmm3;
    xmm15 = m32;
    xmm9 = CT(328 + r12);
    xmm5 = xmm5 - xmm15;
    m24 = xmm5;
    xmm5 = m32;
    xmm2 = xmm2 * xmm5;
    xmm5 = xmm5 * xmm4;
    xmm2 = xmm2 - xmm6;
    xmm6 = xmm7;
    xmm2 = xmm2 + xmm5;
    xmm6 = xmm6 + xmm8;
    xmm12 = m24;
    xmm3 = xmm3 * xmm12;
    xmm4 = xmm4 * xmm12;
    xmm2 = xmm2 + xmm3;
    m32 = xmm6;
    xmm2 = xmm2 + xmm4;
    xmm3 = m32;
    xmm7 = xmm7 - xmm3;
    xmm7 = xmm7 + xmm8;
    m24 = xmm7;
    xmm4 = m32;
    m64 = xmm4;
    ecx = HI16(m64);
    ecx &= 32752;
    ecx >>= 4;
    ecx += edx;
    xmm3 = m24;
    if (s32(ecx) >= 2057)
        goto L67;
    if (s32(ecx) > 1984)
        goto L58;
    m32 = xmm1;
    xmm1 = m32;
    xmm1 = xmm1 + CT(851);
    m32 = xmm1;
    xmm2 = m32;
    xmm2 = xmm2 * xmm0;
    goto L56;
L46:
    m32 = xmm2;
    xmm1 = m32;
    xmm0 = m32;
    (void)xmm0;
    /* divsd 0.0/0.0 */
    xmm1 = as_f64(X86_DEFAULT_NAN);
    m32 = xmm1;
    xmm2 = m32;
    goto L56;
L47:
    if (s32(r13) >= 1023)
        goto L51;
    if (bl == 0)
        goto L56;
    xmm2 = xmm0;
    xmm2 = xmm2 * xmm0;
    goto L56;
L51:
    if (bl == 0)
        goto L53;
    r9 &= r12;
    xmm2 = xmm2 * CT(853 + r9);
    goto L56;
L53:
    xmm3 = xmm3 * xmm3;
    xmm3 = xmm3 * xmm0;
    r9 &= r12;
    xmm2 = xmm3;
    xmm2 = xmm2 * CT(853 + r9);
    goto L56;
L54:
    xmm2 = xmm3;
    xmm2 = sse_addsd(xmm2, xmm0);
    goto L56;
L55:
    xmm3 = sse_addsd(xmm3, xmm0);
    m32 = xmm3;
    m24 = CT(853);
    cl = TOPBYTE(m32) & 0x80;
    bl = TOPBYTE(m24) & 0x7f;
    bl |= cl;
    m24 = as_f64((as_u64(m24) & 0x00ffffffffffffffULL) | ((uint64_t)bl << 56));
    xmm2 = m24;
    xmm0 = m24;
    xmm2 = xmm2 * xmm0;
    goto L56;
L57:
    if (r11 == 0)
        goto L30;
    goto L55;
L58:
    xmm6 = CT(844);
    xmm7 = m64;
    xmm8 = xmm8 + xmm2;
    xmm10 = xmm10 + xmm9;
    xmm2 = xmm2 + xmm3;
    xmm6 = xmm6 * xmm8;
    xmm11 = xmm11 + xmm10;
    xmm6 = xmm6 + CT(843);
    xmm11 = xmm11 + xmm2;
    xmm6 = xmm6 * xmm8;
    xmm9 = xmm11;
    xmm6 = xmm6 + CT(842);
    xmm6 = xmm6 * xmm8;
    xmm6 = xmm6 + CT(841);
    xmm6 = xmm6 * xmm8;
    xmm4 = xmm4 + xmm6;
    m32 = xmm4;
    xmm8 = m32;
    xmm7 = xmm7 - xmm8;
    xmm7 = xmm7 + xmm6;
    m24 = xmm7;
    xmm12 = m32;
    xmm5 = m24;
    xmm9 = xmm9 + xmm12;
    m32 = xmm9;
    xmm10 = m32;
    xmm12 = xmm12 - xmm10;
    xmm12 = xmm12 + xmm11;
    xmm11 = CT(857);
    m24 = xmm12;
    xmm3 = m32;
    xmm11 = xmm11 * xmm3;
    xmm4 = m24;
    m32 = xmm11;
    xmm4 = xmm4 + xmm5;
    xmm13 = m32;
    xmm11 = m16;
    xmm13 = xmm13 - xmm3;
    xmm4 = xmm4 * m16;
    m24 = xmm13;
    xmm6 = xmm11;
    xmm15 = m32;
    xmm14 = m24;
    xmm6 = xmm6 * CT(857);
    xmm15 = xmm15 - xmm14;
    m32 = xmm15;
    xmm2 = m32;
    xmm5 = CT(855);
    xmm3 = xmm3 - xmm2;
    m24 = xmm3;
    xmm3 = m32;
    xmm2 = m24;
    m32 = xmm6;
    xmm7 = m32;
    xmm7 = xmm7 - m16;
    m24 = xmm7;
    xmm9 = m32;
    xmm8 = m24;
    xmm9 = xmm9 - xmm8;
    m32 = xmm9;
    xmm10 = m32;
    xmm11 = xmm11 - xmm10;
    m24 = xmm11;
    xmm6 = m32;
    xmm14 = xmm6;
    xmm14 = xmm14 * xmm3;
    xmm6 = xmm6 * xmm2;
    xmm5 = xmm5 + xmm14;
    xmm15 = m24;
    xmm2 = xmm2 * xmm15;
    xmm3 = xmm3 * xmm15;
    xmm6 = xmm6 + xmm2;
    m32 = xmm5;
    xmm6 = xmm6 + xmm3;
    xmm12 = m32;
    xmm6 = xmm6 + xmm4;
    xmm12 = xmm12 - CT(855);
    m88 = xmm12;
    xmm13 = m88;
    xmm3 = CT(849);
    xmm14 = xmm14 - xmm13;
    m80 = xmm14;
    xmm5 = m80;
    edx = LO32(m32);
    esi = edx;
    edx &= 127;
    xmm5 = xmm5 + xmm6;
    xmm3 = xmm3 * xmm5;
    edx += edx;
    xmm2 = CT(586 + edx - 1);
    esi = sar32(esi, 7);
    xmm3 = xmm3 + CT(848);
    xmm3 = xmm3 * xmm5;
    xmm3 = xmm3 + CT(847);
    xmm3 = xmm3 * xmm5;
    xmm3 = xmm3 + CT(846);
    xmm3 = xmm3 * xmm5;
    xmm3 = xmm3 + CT(845);
    xmm3 = xmm3 * xmm5;
    xmm3 = xmm3 * xmm2;
    xmm3 = xmm3 + CT(586 + edx);
    xmm4 = xmm3;
    xmm4 = xmm4 + xmm2;
    m72 = xmm4;
    ecx = HI16(m72);
    ebx = ecx;
    ebx &= 32752;
    ebx >>= 4;
    edx = esi - 1023 + ebx;
    if (s32(edx) >= 1024)
        goto L66;
    if (s32(edx) < -1022)
        goto L61;
    ecx &= (uint32_t)-32753;
    edx = edx + 1023;
    edx &= 2047;
    edx <<= 4;
    ecx |= edx;
    m72 = set_hi16(m72, ecx);
    xmm2 = m72;
    xmm2 = xmm2 * xmm0;
    m72 = xmm2;
    goto L56;
L61:
    if (s32(edx) < -1032)
        goto L63;
    m32 = xmm4;
    esi += 1223;
    xmm1 = m32;
    esi &= 2047;
    xmm2 = xmm2 - xmm1;
    m64 = CT(853);
    xmm3 = xmm3 + xmm2;
    m24 = xmm3;
    xmm7 = m32;
    xmm2 = CT(857);
    xmm2 = xmm2 * xmm7;
    xmm9 = m24;
    m32 = xmm2;
    xmm3 = m32;
    edx = (uint32_t)(pow_cout_tab[853] >> 48);
    xmm3 = xmm3 - xmm7;
    m24 = xmm3;
    edx &= (uint32_t)-32753;
    xmm5 = m32;
    xmm4 = m24;
    esi <<= 4;
    xmm5 = xmm5 - xmm4;
    m32 = xmm5;
    edx |= esi;
    xmm6 = m32;
    m64 = set_hi16(m64, edx);
    xmm7 = xmm7 - xmm6;
    m24 = xmm7;
    xmm11 = m32;
    xmm12 = m24;
    xmm10 = m64;
    xmm12 = xmm12 + xmm9;
    xmm11 = xmm11 * xmm10;
    xmm12 = xmm12 * xmm10;
    xmm8 = CT(859);
    xmm12 = xmm12 + xmm11;
    xmm0 = xmm0 * xmm8;
    m32 = CT(851);
    xmm14 = m32;
    xmm13 = m32;
    xmm14 = xmm14 * xmm13;
    xmm0 = xmm0 * xmm12;
    m32 = xmm14;
    xmm15 = m32;
    xmm0 = xmm0 + xmm15;
    xmm2 = xmm0;
    m72 = xmm2;
    goto L56;
L63:
    if (s32(edx) < -1084)
        goto L65;
    esi += 1223;
    esi &= 2047;
    edx = (uint32_t)(pow_cout_tab[853] >> 48);
    esi <<= 4;
    edx &= (uint32_t)-32753;
    m64 = xmm1;
    edx |= esi;
    m64 = set_hi16(m64, edx);
    xmm2 = CT(859);
    xmm1 = m64;
    xmm0 = xmm0 * xmm2;
    xmm4 = xmm4 * xmm1;
    m32 = CT(851);
    xmm0 = xmm0 * xmm4;
    xmm4 = m32;
    xmm3 = m32;
    xmm4 = xmm4 * xmm3;
    m32 = xmm4;
    xmm5 = m32;
    xmm0 = xmm0 - xmm5;
    xmm2 = xmm0;
    m72 = xmm2;
    goto L56;
L65:
    m32 = CT(851);
    xmm2 = m32;
    xmm1 = m32;
    xmm2 = xmm2 * xmm1;
    m32 = xmm2;
    xmm3 = m32;
    xmm0 = xmm0 * xmm3;
    xmm2 = xmm0;
    m72 = xmm2;
    goto L56;
L66:
    m32 = CT(850);
    xmm2 = m32;
    xmm1 = m32;
    xmm2 = xmm2 * xmm1;
    m32 = xmm2;
    xmm3 = m32;
    xmm0 = xmm0 * xmm3;
    xmm2 = xmm0;
    m72 = xmm2;
    goto L56;
L67:
    {
        uint32_t dl = TOPBYTE(m64) & 0x80;
        dl >>= 7;
        bl ^= dl;
    }
    xmm2 = CT(850 + bl);
    xmm2 = xmm2 * xmm2;
    xmm2 = xmm2 * xmm0;
    goto L56;
L68:
    if (r9 != 0)
        goto L70;
    if (esi != 0)
        goto L36;
L70:
    r9 &= 1;
    xmm2 = CT(853 + r9);
    goto L56;
L71:
    xmm3 = xmm3 * xmm3;
    if (bl == 0)
        goto L73;
    r9 &= r12;
    xmm2 = CT(853 + r9);
    xmm2 = xmm2 / xmm3;
    goto L56;
L73:
    r9 &= r12;
    xmm2 = CT(853 + r9);
    xmm2 = xmm2 * xmm3;
    goto L56;
L74:
    if (r11 != 0)
        goto L55;
    if (edx != 0)
        goto L30;
    if (r10 == 0)
        goto L30;
    goto L55;
L56:
    (void)m40;
    (void)m48;
    (void)m72;
    (void)m80;
    (void)m88;
    return xmm2;
}

/* ---------------------------------------------------------------------------
 * __svml_pow8_ha, one lane
 * ------------------------------------------------------------------------- */

/* vfpclasspd $223: QNaN, +0, -0, +Inf, -Inf, negative finite, SNaN */
static int fpclass_x_223(uint64_t ux)
{
    return ux - 1 >= DBL_MAX_BITS;
}

/* vfpclasspd $153: QNaN, +Inf, -Inf, SNaN */
static int fpclass_y_153(uint64_t uy)
{
    return (uy & 0x7fffffffffffffffULL) >= POS_INF_BITS;
}

/* vgetmantpd $10 (interval [1/2,1)) and vgetexppd for positive finite nonzero x */
static void getmant_getexp(uint64_t ux, double *mant, double *expon)
{
    int be = (int)(ux >> 52);
    uint64_t frac = ux & 0x000fffffffffffffULL;
    if (be == 0) {
        be = 1;
        while (!(frac & 0x0010000000000000ULL)) {
            frac <<= 1;
            be--;
        }
        frac &= 0x000fffffffffffffULL;
    }
    *mant = as_f64(0x3fe0000000000000ULL | frac);
    *expon = (double)(be - 1023);
}

double optbench_pow(double x, double y)
{
    uint64_t ux = as_u64(x), uy = as_u64(y), idx, idx2, u2;
    double zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10;
    double zmm11, zmm12, zmm14, zmm15;
    double mant, expon, dblrcp, th, tl, the, tle;

    if (fpclass_x_223(ux) || fpclass_y_153(uy))
        return pow_rare(x, y);

    getmant_getexp(ux, &mant, &expon);

    /* DblRcp ~ 1/Mantissa, rounded to 5 fractional bits */
    dblrcp = rcp14_rndscale(mant);

    /* Prepare table index; bit 4 selects the upper 16 entries (vpsllq $59 + vpmovq2m) */
    idx = as_u64(dblrcp) >> 47;
    th = (idx & 16) ? F(pow_log_th[16 + (idx & 15)]) : F(pow_log_th[idx & 15]);
    tl = (idx & 16) ? F(pow_log_tl[16 + (idx & 15)]) : F(pow_log_tl[idx & 15]);

    /* add 1 to Expon if DblRcp<1.5 */
    if (dblrcp < F(K832))
        expon = expon + F(K896);

    /* Mantissa/2 */
    zmm1 = mant * F(K768);
    /* Reduced argument: R = (DblRcp*Mantissa - 1)*0.5 */
    zmm12 = fma(zmm1, dblrcp, -F(K768));
    /* R^2 */
    zmm8 = zmm12 * zmm12;
    zmm9 = fma(F(K1024), zmm12, F(K1088));
    zmm2 = fma(F(K1152), zmm12, F(K1216));
    /* Expon + Th */
    zmm6 = th + expon;
    /* P7_10 */
    zmm9 = fma(zmm8, zmm9, zmm2);
    zmm15 = fma(F(K1536), zmm12, F(K1600));
    zmm7 = fma(F(K1280), zmm12, F(K1344));
    zmm3 = fma(F(K1408), zmm12, F(K1472));
    /* R^4 */
    zmm11 = zmm8 * zmm8;
    /* P3_6 */
    zmm7 = fma(zmm8, zmm7, zmm3);
    /* P3_10 */
    zmm9 = fma(zmm11, zmm9, zmm7);
    /* P1_10 */
    zmm9 = fma(zmm8, zmm9, zmm15);
    /* Tl + R*P1_10 */
    zmm9 = fma(zmm12, zmm9, tl);
    /* Expon + Th + C1H*R_high */
    zmm5 = fma(F(K960), zmm12, zmm6);
    /* Rh = (c1h*R)_high */
    zmm6 = zmm5 - zmm6;
    /* Expon + Th + Rh_high - (R*Rh)_high */
    zmm4 = fma(-zmm12, zmm6, zmm5);
    /* Rl = (C1H*R)_low */
    zmm14 = fma(zmm12, F(K960), -zmm6);
    /* (R*Rh)_high */
    zmm5 = zmm5 - zmm4;
    /* Rl - R*Rl */
    zmm14 = fma(-zmm14, zmm12, zmm14);
    /* (R*Rh)_low */
    zmm12 = fma(zmm6, zmm12, -zmm5);
    /* Rl - R*Rl - (R*Rh)_low */
    zmm7 = zmm14 - zmm12;
    zmm8 = zmm9 + zmm7;
    /* high part of log2(input1) */
    zmm10 = zmm4 + zmm8;
    /* (y*High)_high */
    zmm12 = mul_rz(zmm10, y);
    /* Tlh */
    zmm9 = zmm10 - zmm4;
    /* Zl = (y*_HIGH)_low */
    zmm10 = fma_rz(y, zmm10, -zmm12);
    /* Tll */
    zmm4 = zmm8 - zmm9;
    /* Zl = y*Tll + Zl */
    zmm4 = fma_rz(y, zmm4, zmm10);
    zmm3 = add_rz(zmm12, zmm4);

    /* Filter very large |y*log2(x)| (vcmppd $22: not <=, unordered true) */
    if (!(fabs(zmm3) <= F(K2432)))
        return pow_rare(x, y);

    zmm12 = zmm3 - zmm12;
    zmm2 = add_rd(zmm3, F(K1664));
    /* exp2 computation starts here */
    zmm5 = reduce_m4_rd(zmm3);
    zmm6 = zmm4 - zmm12;
    zmm7 = zmm5 + zmm6;
    /* ensure |R|<2 even for special cases */
    zmm7 = as_f64(as_u64(zmm7) & K1728);
    /* Table lookup: The, Tle/The */
    idx2 = as_u64(zmm2) & 15;
    tle = F(pow_exp_tle[idx2]);
    the = F(pow_exp_the[idx2]);
    /* Re^2 */
    zmm14 = zmm7 * zmm7;
    zmm1 = fma(F(K1792), zmm7, F(K1856));
    zmm11 = fma(F(K1920), zmm7, F(K1984));
    zmm15 = fma(F(K2048), zmm7, F(K2112));
    u2 = (as_u64(zmm2) << 48) & K2496;
    zmm1 = fma(zmm14, zmm1, zmm11);
    zmm1 = fma(zmm14, zmm1, zmm15);
    /* R*poly + The/Tle */
    zmm1 = fma(zmm7, zmm1, tle);
    /* The + The*(R*poly + The/Tle) */
    zmm1 = fma(the, zmm1, the);
    /* scaled result */
    zmm1 = zmm1 * as_f64(u2);
    return zmm1;
}

/* ---------------------------------------------------------------------------
 * NumPy DOUBLE_power with a stride-0 exponent: y in {-1, 0, 0.5, 1, 2} is
 * computed as 1/x, 1, sqrt(x), x, x*x and never reaches the vector routine.
 * NaN results follow the SSE rules of divsd/sqrtsd/mulsd.
 * ------------------------------------------------------------------------- */

static double sse_recip(double v)
{
    return is_nan_bits(v) ? quiet_nan(v) : 1.0 / v;
}

static double sse_sqrt(double v)
{
    if (is_nan_bits(v))
        return quiet_nan(v);
    if (v < 0.0)
        return as_f64(X86_DEFAULT_NAN);
    return sqrt(v);
}

static double sse_square(double v)
{
    return is_nan_bits(v) ? quiet_nan(v) : v * v;
}

void optbench_pow_scalar_array(const double *x, double y, double *out, long n)
{
    long i;
    uint64_t u;

    if (y == -1.0) {
        for (i = 0; i < n; i++)
            out[i] = sse_recip(x[i]);
    } else if (y == 0.0) {
        for (i = 0; i < n; i++)
            out[i] = 1.0;
    } else if (y == 0.5) {
        for (i = 0; i < n; i++)
            out[i] = sse_sqrt(x[i]);
    } else if (y == 1.0) {
        for (i = 0; i < n; i++) {
            memcpy(&u, &x[i], sizeof u);
            memcpy(&out[i], &u, sizeof u);
        }
    } else if (y == 2.0) {
        for (i = 0; i < n; i++)
            out[i] = sse_square(x[i]);
    } else {
        for (i = 0; i < n; i++)
            out[i] = optbench_pow(x[i], y);
    }
}
