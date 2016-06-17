/*
Copyright 2015 Arun Kumar

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package svm_scale;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import org.apache.hadoop.io.Writable;

/**
 *
 * @author arun
 */
public class DataTupleS implements Serializable, Writable {
  //sid key in seqfile; value tuple is (forkey, label, xS)

  public int forkey;
  public double label;
  public double xS[];

  DataTupleS() {
    label = forkey = 0;
    xS = null;
  }

  DataTupleS(int len) {
    label = forkey = 0;
    xS = new double[len];
  }

  DataTupleS(int f, double l, double[] x) {
    forkey = f;
    label = l;
    xS = x;
  }

  @Override
  public String toString() {
    String str = "RDataTuple{forkey = " + forkey + " label = " + label + ", xS = " + csvPrint() + '}';
    return str;
  }

  public String csvPrint() {
    String str = Double.toString(xS[0]);
    for (int col = 1; col < xS.length; col++) {
      str += "," + xS[col];
    }
    return str;
  }

  @Override
  public void write(DataOutput d) throws IOException {
    d.writeInt(forkey);
    d.writeDouble(label);
    d.writeInt(xS.length);
    for (int i = 0; i < xS.length; i++) {
      d.writeDouble(xS[i]);
    }
  }

  //string format: <forkey> <label> <xS as csv>
  public String writeString(int sid) {
    String str = sid + " " + forkey + " " + label + " " + csvPrint();
    return str;
  }

  //string format: <sid> <forkey> <label> <xS as csv>; sid is returned
  public int readString(String str) {
    String[] v = str.split(" ");
    forkey = Integer.parseInt(v[1]);
    label = Double.parseDouble(v[2]);
    String[] xs = v[3].split(",");
    xS = new double[xs.length];
    for (int i = 0; i < xs.length; i++) {
      xS[i] = Double.parseDouble(xs[i]);
    }
    return Integer.parseInt(v[0]);
  }

  @Override
  public void readFields(DataInput di) throws IOException {
    forkey = di.readInt();
    label = di.readDouble();
    int len = di.readInt();
    xS = new double[len];
    for (int i = 0; i < len; i++) {
      xS[i] = di.readDouble();
    }
  }
}
