package org.ogolem.optbench;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import org.ogolem.core.BondInfo;
import org.ogolem.core.CartesianFullBackend;
import org.ogolem.core.Gradient;

/**
 * Cartesian backend that evaluates the OptBench harness potential.
 *
 * <p>OGOLEM stores cartesians in bohr. Collision radii in {@code AtomicProperties} are bohr and
 * the argon entry is 1.85, sized for the argon Lennard-Jones sigma below. The harness potential
 * is reduced (sigma 1). This class divides internal bohr by that sigma before the socket call
 * and divides the returned gradient by the same sigma. One socket line is one harness entry.
 */
public final class OptbenchBackend implements CartesianFullBackend {

  private static final long serialVersionUID = 20261018L;

  /** {@code AtomicProperties} ljSigmaTable value for Ar, in bohr. */
  static final double AR_SIGMA_BOHR = 6.434516991;

  private static final int SOCKET_TIMEOUT_MS = 180_000;

  private transient Socket socket;
  private transient BufferedReader reader;
  private transient BufferedWriter writer;

  public OptbenchBackend() {}

  @Override
  public OptbenchBackend copy() {
    return new OptbenchBackend();
  }

  @Override
  public String getMethodID() {
    return "OptBench harness potential";
  }

  @Override
  public double energyCalculation(
      long lID,
      int iIteration,
      double[] xyz1D,
      String[] saAtomTypes,
      short[] atomNos,
      int[] atsPerMol,
      double[] energyparts,
      int iNoOfAtoms,
      float[] faCharges,
      short[] spins,
      BondInfo bonds,
      boolean hasRigidEnvironment) {
    double[] unused = new double[3 * iNoOfAtoms];
    return evaluate(xyz1D, iNoOfAtoms, energyparts, unused, false);
  }

  @Override
  public void gradientCalculation(
      long lID,
      int iIteration,
      double[] xyz1D,
      String[] saAtomTypes,
      short[] atomNos,
      int[] atsPerMol,
      double[] energyparts,
      int iNoOfAtoms,
      float[] faCharges,
      short[] spins,
      BondInfo bonds,
      Gradient gradient,
      boolean hasRigidEnvironment) {
    double[][] matrix = gradient.getTotalGradient();
    if (matrix == null || matrix.length != 3 || matrix[0].length < iNoOfAtoms) {
      throw new IllegalStateException("OGOLEM gradient storage is not a 3 by atom matrix");
    }
    gradient.zeroGradient();
    double[] flat = new double[3 * iNoOfAtoms];
    double energy = evaluate(xyz1D, iNoOfAtoms, energyparts, flat, true);
    for (int atom = 0; atom < iNoOfAtoms; atom++) {
      matrix[0][atom] = flat[3 * atom];
      matrix[1][atom] = flat[3 * atom + 1];
      matrix[2][atom] = flat[3 * atom + 2];
    }
    gradient.setTotalEnergy(energy);
  }

  private double evaluate(
      double[] xyzBohr, int atoms, double[] energyparts, double[] gradientBohr, boolean wantGradient) {
    if (xyzBohr.length < 3 * atoms) {
      throw new IllegalArgumentException("coordinate array is shorter than the atom count");
    }
    try {
      connect();
      StringBuilder line = new StringBuilder(24 + atoms * 72);
      line.append("calculate ").append(atoms);
      for (int atom = 0; atom < atoms; atom++) {
        append(line, xyzBohr[atom] / AR_SIGMA_BOHR);
        append(line, xyzBohr[atoms + atom] / AR_SIGMA_BOHR);
        append(line, xyzBohr[2 * atoms + atom] / AR_SIGMA_BOHR);
      }
      writer.write(line.toString());
      writer.write('\n');
      writer.flush();
      String reply = reader.readLine();
      if (reply == null) {
        throw new IOException("harness server closed the socket");
      }
      String[] parts = reply.trim().split("\\s+");
      if (parts.length == 0) {
        throw new IOException("empty harness reply");
      }
      if (parts[0].equals("cap")) {
        System.exit(3);
      }
      if (parts[0].equals("found")) {
        double energy = Double.parseDouble(parts[2]);
        fillParts(energyparts, energy);
        System.exit(0);
      }
      if (!parts[0].equals("ok") || parts.length < 3 + (wantGradient ? 3 * atoms : 0)) {
        throw new IOException("harness reply: " + reply);
      }
      double energy = Double.parseDouble(parts[2]);
      fillParts(energyparts, energy);
      if (wantGradient) {
        for (int atom = 0; atom < atoms; atom++) {
          int base = 3 + 3 * atom;
          // r_reduced = r_bohr / AR_SIGMA_BOHR, so dE/d(bohr) = dE/d(reduced) / AR_SIGMA_BOHR.
          gradientBohr[3 * atom] = Double.parseDouble(parts[base]) / AR_SIGMA_BOHR;
          gradientBohr[3 * atom + 1] = Double.parseDouble(parts[base + 1]) / AR_SIGMA_BOHR;
          gradientBohr[3 * atom + 2] = Double.parseDouble(parts[base + 2]) / AR_SIGMA_BOHR;
        }
      }
      return energy;
    } catch (IOException exc) {
      throw new IllegalStateException("harness potential call failed", exc);
    }
  }

  private static void append(StringBuilder line, double value) {
    line.append(' ');
    line.append(String.format(Locale.US, "%.16e", value));
  }

  private static void fillParts(double[] energyparts, double energy) {
    if (energyparts == null) {
      return;
    }
    for (int i = 0; i < energyparts.length; i++) {
      energyparts[i] = 0.0;
    }
    if (energyparts.length > 0) {
      energyparts[0] = energy;
    }
  }

  private void connect() throws IOException {
    if (socket != null && socket.isConnected() && !socket.isClosed()) {
      return;
    }
    String endpoint = System.getenv("OPTBENCH_POTSERVER");
    if (endpoint == null || !endpoint.contains(":")) {
      throw new IOException("OPTBENCH_POTSERVER must be host:port");
    }
    int colon = endpoint.lastIndexOf(':');
    String host = endpoint.substring(0, colon);
    int port = Integer.parseInt(endpoint.substring(colon + 1));
    Socket opened = new Socket();
    opened.setTcpNoDelay(true);
    opened.connect(new InetSocketAddress(host, port), SOCKET_TIMEOUT_MS);
    opened.setSoTimeout(SOCKET_TIMEOUT_MS);
    BufferedWriter out =
        new BufferedWriter(new OutputStreamWriter(opened.getOutputStream(), StandardCharsets.UTF_8));
    BufferedReader in =
        new BufferedReader(new InputStreamReader(opened.getInputStream(), StandardCharsets.UTF_8));
    out.write("hello\n");
    out.flush();
    String ready = in.readLine();
    if (ready == null || !ready.equals("ok")) {
      opened.close();
      throw new IOException("harness server hello failed: " + ready);
    }
    socket = opened;
    writer = out;
    reader = in;
  }
}
