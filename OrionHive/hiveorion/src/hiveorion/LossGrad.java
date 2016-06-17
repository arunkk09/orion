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

package hiveorion;

import java.util.ArrayList;
import org.apache.hadoop.hive.ql.exec.UDAF;
import org.apache.hadoop.hive.ql.exec.UDAFEvaluator;

/**
 *
 * @author arun
 */
public class LossGrad extends UDAF {

  /**
   * The actual class for doing the aggregation. Hive will automatically look
   * for all internal classes of the UDAF that implements UDAFEvaluator.
   */
  public static class LossGradEvaluator implements UDAFEvaluator {

    ArrayList<Double> lossgrad; //loss, followed by grad vec

    public LossGradEvaluator() {
      super();
      lossgrad = new ArrayList<Double>();
      lossgrad.add(0.0); //loss
    }

    /**
     * Reset the state of the aggregation.
     */
    public void init() {
      lossgrad.set(0, 0.0);
    }

    double sigma(double v) {
      if (v > 30) {
        return 1.0 / (1.0 + Math.exp(-v));
      } else {
        return Math.exp(v) / (1.0 + Math.exp(v));
      }
    }

    /**
     * Iterate through one row of original data.
     *
     * This UDF accepts arbitrary number of String arguments, so we use
     * String[]. If it only accepts a single String, then we should use a single
     * String argument.
     *
     * This function should always return true.
     */
    //args: y, x, w
    public boolean iterate(Double y, ArrayList<Double> x, ArrayList<Double> w) {
      if ((lossgrad != null) && (lossgrad.size() == 1)) {
        for (int i = 0; i < w.size(); i++) {
          lossgrad.add(0.0); //for grad
        }
      }
      //dot product, scale, add into loss, scale and add into grad
      double wx = 0.0;
      for (int i = 0; i < w.size(); i++) {
        wx += w.get(i) * x.get(i);
      }
      double sig = sigma(-y * wx);
      double c = -y * sig; // scale factor
      //scale and add into grad
      for (int i = 0; i < w.size(); i++) {
        lossgrad.set(1 + i, lossgrad.get(1 + i) + c * x.get(i));
      }
      //add into loss
      lossgrad.set(0, lossgrad.get(0) + Math.log(1 + Math.exp(-y * wx)));
      return true;
    }

    /**
     * Terminate a partial aggregation and return the state.
     */
    public ArrayList<Double> terminatePartial() {
      ArrayList<Double> ret = new ArrayList<Double>();
      for (int i = 0; i < lossgrad.size(); i++) {
        ret.add(lossgrad.get(i));
      }
      return ret;
    }

    /**
     * Merge with a partial aggregation.
     *
     * This function should always have a single argument which has the same
     * type as the return value of terminatePartial().
     *
     * This function should always return true.
     */
    public boolean merge(ArrayList<Double> o) {
      lossgrad.set(0, lossgrad.get(0) + o.get(0));//loss is sum of sums
      if (lossgrad.size() < 2) {
        for (int i = 1; i < o.size(); i++) {
          lossgrad.add(o.get(i));
        }
      } else {
        for (int i = 1; i < o.size(); i++) {
          lossgrad.set(i, lossgrad.get(i) + o.get(i)); //grad is also sum of sums
        }
      }
      return true;
    }

    /**
     * Terminates the aggregation and return the final result.
     */
    public ArrayList<Double> terminate() {
      return lossgrad;
    }
  }
}
