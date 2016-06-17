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

#ifndef MYHASH
#define MYHASH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//Jenkins hash function from Wikipedia!
//the hash id key is assumed to be an integer that is 4B long; returns the bkt num (0 to numbkts - 1)
unsigned int hash_jen(unsigned int key, unsigned int numbkts) {
	unsigned int hash, i;
	for(hash = i = 0; i < 4; ++i) {
		hash += ((char*)(&key))[i];
		hash += (hash << 10);
		hash ^= (hash >> 6);
	}
	hash += (hash << 3);
	hash ^= (hash >> 11);
	hash += (hash << 15);
	return hash % numbkts;
}

//insert (key,v1,v2) into the hash; assumption is that this key is unique (no repetitions)
//if collision occurs, we do linear search to insert and ret false; else ret true
bool add_entry(unsigned int key, double v1, double v2, double* thehash, unsigned int numbkts) {
	unsigned int hbkt = hash_jen(key, numbkts);
	unsigned int idthere = (unsigned int)thehash[hbkt*3];
	bool ret = true;
	if(idthere != 0) { //collision
		ret = false;
		while(idthere != 0) { //will get free bkt eventually
			//printf("\t Collision for key %d at bkt %d which already had key %d\n", key, hbkt, idthere);
			hbkt = (hbkt + 1) % numbkts;
			idthere = (unsigned int)thehash[hbkt*3];
		}
	}
	//printf("Adding keyid %d at bkt %d\n", key, hbkt);
	thehash[hbkt*3] = (double) key;
	thehash[hbkt*3 + 1] = v1;
	thehash[hbkt*3 + 2] = v2;
	return ret;
}

//find bkt of given key in the hash and store in bkt; return true if entry found, false o/w
//after hashing, we need to match key; if collision occured, we need to do linear search and loopback
bool find_bkt(unsigned int key, double* thehash, unsigned int numbkts, unsigned int *bkt) {
	unsigned int hbkt = hash_jen(key, numbkts);
	unsigned int idthere = (unsigned int)thehash[hbkt*3];
	if(idthere == 0) { //bkt is empty; key not present
		bkt = NULL;
		//elog(WARNING, "find_bkt: key %d got bkt %d that was empty\n", key, hbkt);
		return false;
	}
	//bkt not empty, but key not matched; linear search on till id matches
	unsigned int tbkt = hbkt;
	unsigned int cnt = 0;
	while(idthere != key) {
		cnt++;
		if(cnt >= numbkts) {
			//elog(WARNING, "find_bkt: key %d saw cnt %d bkts out of numbkts %d and still no match\n", key, cnt, numbkts);
			break;
		}
		tbkt = (tbkt + 1) % numbkts;
		idthere = (unsigned int)thehash[tbkt*3];
	}
	if(idthere != key) { //did not find a match on idkey with same hash
		bkt = NULL;
		//elog(WARNING, "find_bkt: key %d did not find match in entire hash\n", key);
		return false;
	}
	//found match at tbkt; return it!
	*bkt = tbkt;
	return true;
}

#endif
