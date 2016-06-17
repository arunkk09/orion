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
public class FSQLPlus2 extends UDAF {

  /**
   * The actual class for doing the aggregation. Hive will automatically look
   * for all internal classes of the UDAF that implements UDAFEvaluator.
   */
  public static class FSQLPlus2Evaluator implements UDAFEvaluator {

    ArrayList<Double> lossgrad; //loss, sum of G, partial grad vec (length d1-1)

    public FSQLPlus2Evaluator() {
      super();
      lossgrad = new ArrayList<Double>();
      lossgrad.add(0.0); //loss
      lossgrad.add(0.0); //sum of G
    }

    /**
     * Reset the state of the aggregation.
     */
    public void init() {
      for (int i = 0; i < lossgrad.size(); i++) {
        lossgrad.set(i, 0.0);
      }
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
    //args: pdp wRxR from HR, y, xS, wS
    //aggstate: loss, sum of G, partial grad vec (length d1-1)
    public boolean iterate(Double wRxR, Double y, ArrayList<Double> xS, ArrayList<Double> wS) {
      if ((lossgrad != null) && (lossgrad.size() == 2)) {
        int len = wS.size();
        for (int i = 0; i < len; i++) {
          lossgrad.add(0.0); //for grad
        }
      }
      //compute partial dot products, add with wRxR to get full dot pdt
      double fulldot = wRxR;
      for (int i = 0; i < wS.size(); i++) {
        fulldot += wS.get(i) * xS.get(i);
      }
      double gwx = -y * sigma(-y * fulldot); //scale factor g(w.x)
      //update loss
      lossgrad.set(0, lossgrad.get(0) + Math.log(1 + Math.exp(-y * fulldot)));
      //update v2 with sum of g(w.x); automatic group by sum
      lossgrad.set(1, lossgrad.get(1) + gwx);
      //add to first portion of gradient by scaling xS by gwx; automatic group by sum
      for (int i = 0; i < wS.size(); i++) {
        lossgrad.set(2 + i, lossgrad.get(2 + i) + gwx * xS.get(i));
      }
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
    //aggstate: loss, sum of G, partial grad vec (length d1-1)
    public boolean merge(ArrayList<Double> o) {
      lossgrad.set(0, lossgrad.get(0) + o.get(0));//loss is sum of sums
      lossgrad.set(1, lossgrad.get(1) + o.get(1));//sum of scales is sum of sums
      if (lossgrad.size() <= 2) {
        for (int i = 2; i < o.size(); i++) {
          lossgrad.add(o.get(i));
        }
      } else {
        for (int i = 2; i < o.size(); i++) {
          lossgrad.set(i, lossgrad.get(i) + o.get(i)); //sum of G and grad are also sum of sums
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
