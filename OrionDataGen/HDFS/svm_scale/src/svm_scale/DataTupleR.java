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
public class DataTupleR implements Serializable, Writable {
  //rid key in seqfile; value is xR
  
  public double xR[];

  DataTupleR() {
    xR = null;
  }

  DataTupleR(int len) {
    xR = new double[len];
  }

  DataTupleR(double[] x) {
    xR = x;
  }

  @Override
  public String toString() {
    String str = "DataTupleR{xR = " + csvPrint() + '}';
    return str;
  }

  public String csvPrint() {
    String str = Double.toString(xR[0]);
    for (int col = 1; col < xR.length; col++) {
      str += "," + xR[col];
    }
    return str;
  }
  
    //string format: <xR as csv>
  public String writeString(int rid) {
    String str = rid + " " + csvPrint();
    return str;
  }

  //string format: <rid> <xR as csv>; rid is returned
  public int readString(String str) {
    String[] v = str.split(" ");
    String[] xr = v[1].split(",");
    xR = new double[xs.length];
    for (int i = 0; i < xr.length; i++) {
      xR[i] = Double.parseDouble(xr[i]);
    }
    return Integer.parseInt(v[0]);
  }

  @Override
  public void write(DataOutput d) throws IOException {
    d.writeInt(xR.length);
    for (int i = 0; i < xR.length; i++) {
      d.writeDouble(xR[i]);
    }
  }

  @Override
  public void readFields(DataInput di) throws IOException {
    int len = di.readInt();
    xR = new double[len];
    for (int i = 0; i < len; i++) {
      xR[i] = di.readDouble();
    }
  }
}
