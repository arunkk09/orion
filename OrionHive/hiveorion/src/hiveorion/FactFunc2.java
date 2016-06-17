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
public class FactFunc2 extends UDAF {

  /**
   * The actual class for doing the aggregation. Hive will automatically look
   * for all internal classes of the UDAF that implements UDAFEvaluator.
   */
  public static class FactFunc2Evaluator implements UDAFEvaluator {

    ArrayList<Double> lossgrad; //loss, nb, ndims, followed by grad vec, followed by hashtable with (rid, v1, v2)

    public FactFunc2Evaluator() {
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
    //args: forkey, y, xS, wS, nb, ndims, hashtable with nb and (rid, v1, v2)
    //aggstate: loss, nb, ndims, gradvec, hashtable
    //nb is the number of hash buckets in the hash table, i.e., 1.4*nR
    public boolean iterate(Integer forkey, Double y, ArrayList<Double> xS, ArrayList<Double> wS, Integer nb, Integer ndims, ArrayList<Double> htable) {
      if ((lossgrad != null) && (lossgrad.size() == 1)) {
        lossgrad.add((double) nb);
        lossgrad.add((double) ndims);
        for (int i = 0; i < ndims; i++) {
          lossgrad.add(0.0); //for grad
        }
        for (int i = 0; i < nb; i++) { //the htable is input after func 1 outputs it; its 0th elem is nb
          lossgrad.add(htable.get(1 + 3*i));
          lossgrad.add(htable.get(1 + 3*i + 1));
          lossgrad.add(htable.get(1 + 3*i + 2));
        }
      }
      //compute partial dot products, probe into hash table and get full dot pdt
      int hashloc = 3 + ndims;
      int fndbkt = MyHash.find_bkt(forkey, lossgrad, hashloc, nb);
      if (fndbkt == -1) {
        System.out.println("In FactFunc2, find_bkt failed on forkey " + forkey);
      }
      int fsid = (int) lossgrad.get(hashloc + 3 * fndbkt).doubleValue();
      if (forkey != fsid) {
        System.out.println("In FactFunc2, found bucket has mismatched rid " + fsid + " instead of forkey " + forkey);
      }
      double fulldot = lossgrad.get(hashloc + 3 * fndbkt + 1);
      for (int i = 0; i < wS.size(); i++) {
        fulldot += wS.get(i) * xS.get(i);
      }
      double gwx = -y * sigma(-y * fulldot); //scale factor g(w.x)
      //update v2 with sum of g(w.x); automatic group by
      lossgrad.set(hashloc + 3 * fndbkt + 2, lossgrad.get(hashloc + 3 * fndbkt + 2) + gwx);
      //add to first portion of gradient by scaling xS by gwx
      for (int i = 0; i < wS.size(); i++) {
        lossgrad.set(3 + i, lossgrad.get(3 + i) + gwx * xS.get(i));
      }
      lossgrad.set(0, lossgrad.get(0) + Math.log(1 + Math.exp(-y * fulldot)));
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
          lossgrad.add(o.get(i)); //this already copies the whole hash table with all the rid keys already present! No need to add_entry now! Just (GROUP BY) SUM for v2!
        }
      } else {
        int nb = (int) o.get(1).doubleValue();
        int ndims = (int) o.get(2).doubleValue();
        assert (nb == (int) lossgrad.get(1).doubleValue());
        assert (ndims == (int) lossgrad.get(2).doubleValue());
        for (int i = 3; i < (ndims + 3); i++) {
          lossgrad.set(i, lossgrad.get(i) + o.get(i)); //grad is also sum of sums
        }
        //iterate through hash table and update each v2 by simply adding - it is a group by sum!
        int hashloc = 3 + ndims;
        for (int i = 0; i < nb; i++) {
          lossgrad.set(hashloc + 3 * i + 2, lossgrad.get(hashloc + 3 * i + 2) + o.get(hashloc + 3 * i + 2));
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
