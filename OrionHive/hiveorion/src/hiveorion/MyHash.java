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

/**
 *
 * @author arun
 */
public class MyHash {

  static int hash_jen(int key, int numbkts) {
    int hash = 0, i = 0;
    hash += key & 0x000000ff;
    hash += (hash << 10);
    hash ^= (hash >>> 6);
    hash += key & 0x0000ff00;
    hash += (hash << 10);
    hash ^= (hash >>> 6);
    hash += key & 0x00ff0000;
    hash += (hash << 10);
    hash ^= (hash >>> 6);
    hash += key & 0xff000000;
    hash += (hash << 10);
    hash ^= (hash >>> 6);
    /*for(hash = i = 0; i < 4; ++i) {
     hash += ((char*)(&key))[i];
     hash += (hash << 10);
     hash ^= (hash >> 6);
     }*/
    hash += (hash << 3);
    hash ^= (hash >>> 11);
    hash += (hash << 15);
    return Math.abs(hash % numbkts);
  }

  //insert (key,v1,v2) into the hash; assumption is that this key is unique (no repetitions)
  //if collision occurs, we do linear search to insert and ret false; else ret true
  static boolean add_entry(int key, double v1, double v2, ArrayList<Double> state, int hashloc, int numbkts) {
    int hbkt = hash_jen(key, numbkts);
    int idthere = (int) state.get(hashloc + hbkt * 3).doubleValue();
    boolean ret = true;
    if (idthere != 0) { //collision
      ret = false;
      while (idthere != 0) { //will get free bkt eventually
        //printf("\t Collision for key %d at bkt %d which already had key %d\n", key, hbkt, idthere);
        hbkt = (hbkt + 1) % numbkts;
        idthere = (int) state.get(hashloc + hbkt * 3).doubleValue();
      }
    }
    //printf("Adding keyid %d at bkt %d\n", key, hbkt);
    state.set(hashloc + hbkt * 3, (double) key);
    state.set(hashloc + hbkt * 3 + 1, v1);
    state.set(hashloc + hbkt * 3 + 2, v2);
    return ret;
  }

  //find bkt of given key in the hash and store in bkt; return true bkt if entry found, -1 o/w
  //after hashing, we need to match key; if collision occured, we need to do linear search and loopback
  static int find_bkt(int key, ArrayList<Double> state, int hashloc, int numbkts) { //unsigned int *bkt is ret
    int hbkt = hash_jen(key, numbkts);
    int idthere = (int) state.get(hashloc + hbkt * 3).doubleValue();
    if (idthere == 0) { //bkt is empty; key not present
      //bkt = NULL
      //elog(WARNING, "find_bkt: key %d got bkt %d that was empty\n", key, hbkt);
      return -1;
    }
    //bkt not empty, but key not matched; linear search on till id matches
    int tbkt = hbkt;
    int cnt = 0;
    while (idthere != key) {
      cnt++;
      if (cnt >= numbkts) {
        //elog(WARNING, "find_bkt: key %d saw cnt %d bkts out of numbkts %d and still no match\n", key, cnt, numbkts);
        break;
      }
      tbkt = (tbkt + 1) % numbkts;
      idthere = (int) state.get(hashloc + tbkt * 3).doubleValue();
    }
    if (idthere != key) { //did not find a match on idkey with same hash
      //bkt = NULL;
      //elog(WARNING, "find_bkt: key %d did not find match in entire hash\n", key);
      return -1;
    }
    //found match at tbkt; return it!
    // * bkt = tbkt;
    return tbkt;
  }

  //what is the number of hashes needed to find the given keyid
  static int find_bkt_cnt(int key, ArrayList<Double> state, int hashloc, int numbkts) {
    int cnt = 0;
    int hbkt = hash_jen(key, numbkts);
    cnt++;
    int idthere = (int) state.get(hashloc + hbkt * 3).doubleValue();
    if (idthere == 0) { //bkt is empty; key not present
      return cnt;
    }
    //bkt not empty, but key not matched; search on till id matches or hash differs
    int tbkt = hbkt;
    while (idthere != key) {
      tbkt = (tbkt + 1) % numbkts;
      idthere = (int) state.get(hashloc + tbkt * 3).doubleValue();
      cnt++;
    }
    return cnt;
  }
}
