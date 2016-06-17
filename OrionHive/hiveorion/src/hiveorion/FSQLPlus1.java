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
//this is for R to HR by pdps
public final class FSQLPlus1 extends UDF {

  public Double evaluate(final ArrayList<Double> wR, ArrayList<Double> xR) {
    double ret = 0.0;
    for (int i = 0; i < wR.size(); i++) {
      ret += wR.get(i) * xR.get(i);
    }
    return ret;
  }
}
