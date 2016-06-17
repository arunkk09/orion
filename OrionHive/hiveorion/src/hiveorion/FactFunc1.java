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
public class FactFunc1 extends UDAF {

  /**
   * The actual class for doing the aggregation. Hive will automatically look
   * for all internal classes of the UDAF that implements UDAFEvaluator.
   */
  public static class FactFunc1Evaluator implements UDAFEvaluator {

    ArrayList<Double> htable; //nb, followed by hashtable with (rid, v1, v2)

    public FactFunc1Evaluator() {
      super();
      htable = new ArrayList<Double>();
      htable.add(0.0); //nb
    }

    /**
     * Reset the state of the aggregation.
     */
    public void init() {
      htable.set(0, 0.0);
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
    //args: rid, xR, wR, nb
    //aggstate: nb, hashtable
    public boolean iterate(Integer rid, ArrayList<Double> xR, ArrayList<Double> wR, Integer nb) {
      if ((htable != null) && (htable.size() == 1)) {
        htable.set(0, (double) nb);
        for (int i = 0; i < nb; i++) {
          htable.add(0.0); //for key (hashed rid % nb)
          htable.add(0.0); //for v1 (pdps)
          htable.add(0.0); //for v2 (sum of scaled dps)
        }
      }
      //partial dot product, insert into hash table
      double wRxR = 0.0;
      for (int i = 0; i < wR.size(); i++) {
        wRxR += wR.get(i) * xR.get(i);
      }
      MyHash.add_entry(rid, wRxR, 0.0, htable, 1, nb);
      return true;
    }

    /**
     * Terminate a partial aggregation and return the state.
     */
    public ArrayList<Double> terminatePartial() {
      ArrayList<Double> ret = new ArrayList<Double>();
      for (int i = 0; i < htable.size(); i++) {
        ret.add(htable.get(i));
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
      if (htable.size() == 1) {
        for (int i = 1; i < o.size(); i++) {
          htable.add(o.get(i));
        }
      } else {
        int nb = (int) o.get(0).doubleValue();
        assert(nb == (int) htable.get(0).doubleValue());
        //to merge the hashtables, iterate through o and insert each entry into htable
        for (int i = 0; i < nb; i++) {
          MyHash.add_entry((int)o.get(1 + 3*i).doubleValue(), o.get(1 + 3*i + 1).doubleValue(), 0.0, htable, 1, nb);
        }
      }
      return true;
    }

    /**
     * Terminates the aggregation and return the final result.
     */
    public ArrayList<Double> terminate() {
      return htable;
    }
  }
}
