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
import org.apache.hadoop.hive.ql.exec.UDF;

/**
 *
 * @author arun
 */
public final class ConcatArray extends UDF {

  public ArrayList<Double> evaluate(final ArrayList<Double> l1, ArrayList<Double> l2) {
    if (l1 == null) {
      return l2;
    } else if (l2 == null) {
      return l1;
    } else {
      ArrayList<Double> ret = new ArrayList<Double>();
      for (Double l : l1) {
        ret.add(l);
      }
      for (Double l : l2) {
        ret.add(l);
      }
      return ret;
    }
  }
}
