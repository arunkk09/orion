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
public class FactFunc3 extends UDAF {

  /**
   * The actual class for doing the aggregation. Hive will automatically look
   * for all internal classes of the UDAF that implements UDAFEvaluator.
   */
  public static class FactFunc3Evaluator implements UDAFEvaluator {

    ArrayList<Double> lossgrad; //loss, nb, ndims, followed by grad vec, followed by hashtable with (rid, v1, v2)

    public FactFunc3Evaluator() {
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
    //args: rid, xR, nb, ndims, lossgrad from FactFunc2
    //aggstate: loss, nb, ndims, gradvec, hashtable
    public boolean iterate(Integer rid, ArrayList<Double> xR, Integer nb, Integer ndims, ArrayList<Double> prevlossgrad) {
      /*if ((lossgrad != null) && (lossgrad.size() == 1)) {
        lossgrad.clear();
        lossgrad.addAll(prevlossgrad);
      }*/ //replaced on Nov 7 with element-wise insertion below
      if ((lossgrad != null) && (lossgrad.size() == 1)) {
        lossgrad.add((double) nb);
        lossgrad.add((double) ndims);
        for (int i = 0; i < ndims; i++) {
          lossgrad.add(0.0); //for grad
        }
        for (int i = 0; i < nb; i++) { //the htable is in lossgrad func 2 outputs it; it starts at 3 + ndims
          lossgrad.add(htable.get(3 + ndims + 3*i));
          lossgrad.add(htable.get(3 + ndims + 3*i + 1));
          lossgrad.add(htable.get(3 + ndims + 3*i + 2));
        }
      } //added on Nov 7
      //probe into hash table and get full dot pdt
      int hashloc = 3 + ndims;
      int fndbkt = MyHash.find_bkt(rid, lossgrad, hashloc, nb);
      if (fndbkt == -1) {
        System.out.println("In FactFunc3, find_bkt failed on rid " + rid);
      }
      int fsid = (int) lossgrad.get(hashloc + 3 * fndbkt).doubleValue();
      if (rid != fsid) {
        System.out.println("In FactFunc3, found bucket has mismatched fsid " + fsid + " instead of rid " + rid);
      }
      //add to second portion of gradient by scaling xR by v2 from hash
      int off = 3 + (ndims - xR.size());
      double c = lossgrad.get(hashloc + 3 * fndbkt + 2);
      for (int i = 0; i < xR.size(); i++) {
        lossgrad.set(off + i, lossgrad.get(off + i) + c * xR.get(i));
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
    //aggstate: loss, nb, ndims, gradvec, hashtable
    public boolean merge(ArrayList<Double> o) {
      lossgrad.set(0, lossgrad.get(0) + o.get(0));//loss is sum of sums
      if (lossgrad.size() < 2) {
        for (int i = 1; i < o.size(); i++) {
          lossgrad.add(o.get(i));
        }
      } else {
        int nb = (int) o.get(1).doubleValue();
        int ndims = (int) o.get(2).doubleValue();
        assert (nb == (int) lossgrad.get(1).doubleValue());
        assert (ndims == (int) lossgrad.get(2).doubleValue());
        for (int i = 3; i < (ndims + 3); i++) {
          lossgrad.set(i, lossgrad.get(i) + o.get(i)); //grad is also sum of sums
        }
        //ignore the hashtables
      }
      return true;
    }

    /**
     * Terminates the aggregation and return the final result.
     */
    //return only the loss, gradvec
    public ArrayList<Double> terminate() {
      ArrayList<Double> ret = new ArrayList<Double>();
      ret.add(lossgrad.get(0)); //loss
      int ndims = (int)lossgrad.get(2).doubleValue();
      for (int i = 3; i < (ndims + 3); i++) {
        ret.add(lossgrad.get(i));
      }
      return ret;
    }
  }
}
