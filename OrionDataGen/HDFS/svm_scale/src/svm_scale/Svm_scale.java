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

import java.io.DataOutput;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Random;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

/**
 *
 * @author arun
 */
public class Svm_scale {

  /**
   * Writes out the files for S and R in space-separated-values format
   * S is the entity table (the outer table for the join)
   * Schema of S is (SID Target ForeignKey XS)
   * R is the attribute table (the inner table for the join)
   * Schema of R is (RID XR)
   * nS is number of tuples in S, nR is number of tuples in R
   * dS is number of features in S (including target), dR is number of features in R
   * Variance is a parameter for the Gaussian random number generator (1.0 is recommended)
   * Sfile is the prefix of the name for S files
   * Rfile is the name of the R file
   * The last argument enables generation of only R file or only S files
   */
  public static void main(String[] args) throws IOException {
    if (args.length < 8) {
      System.out.println("Usage: hadoop jar svm_scale.jar svm_scale.Svm_scale <Ratio of nS:nR> <nR> <dS> <dR> <Variance> <0 for binary | 1 for txt> <Sfile> <Rfile> [1 for R alone | 2 for S alone]");
      System.exit(1);
    }

    int r = Integer.parseInt(args[0]);
    int nR = Integer.parseInt(args[1]);
    int dS = Integer.parseInt(args[2]); //includes target
    int dR = Integer.parseInt(args[3]);
    double var = Integer.parseInt(args[4]);
    System.out.println("r " + r + " nR " + nR + " dS " + dS + " dR " + dR + " var " + var);

    int dim = dS + dR - 1;
    int wlenr = dS - 1;
    double[] w = new double[dim];
    DataTupleS si = new DataTupleS(wlenr);
    DataTupleR ri = new DataTupleR(dR);
    double normsq = 0.0;
    Random myrand = new Random();
    myrand.setSeed(0xDEADBEEF);

    int onlyflag = 0;
    if (args.length > 8) {
      onlyflag = Integer.parseInt(args[8]);
    }

    // generate the model
    for (int i = dim - 1; i >= 0; i--) {
      w[i] = myrand.nextDouble() - 0.5;
      normsq += w[i] * w[i];
    }
    double norm = Math.sqrt(normsq);
    for (int i = dim - 1; i >= 0; i--) {
      w[i] /= norm;
    }

    int format = Integer.parseInt(args[5]); //0 for binary | 1 for text
    Path pathS = new Path(args[6]);
    Path pathRdir = new Path("text" + pathS.toString());
    Path pathR = new Path(args[7]);
    Path pathSdir = new Path("text" + pathR.toString());
    Configuration conf = new Configuration();

    FileSystem fs = FileSystem.get(conf);
    if (fs.exists(pathS)) {
      fs.delete(pathS, true);
    }
    if (fs.exists(pathR)) {
      fs.delete(pathR, true);
    }
    if (fs.exists(pathRdir)) {
      fs.delete(pathRdir, true);
    }
    if (fs.exists(pathSdir)) {
      fs.delete(pathSdir, true);
    }

    IntWritable sidkey = new IntWritable();
    IntWritable ridkey = new IntWritable();
    SequenceFile.Writer writerS = null;
    SequenceFile.Writer writerR = null;
    if (format == 0) {
      if (onlyflag != 1) {
        writerS = SequenceFile.createWriter(fs, conf, pathS, sidkey.getClass(), si.getClass());
      }
      if (onlyflag != 2) {
        writerR = SequenceFile.createWriter(fs, conf, pathR, ridkey.getClass(), ri.getClass());
      }
    }
    FSDataOutputStream textS = null;
    FSDataOutputStream textR = null;
    if (format == 1) {
      if (onlyflag != 1) {
        //need to create it in its own dir
        fs.mkdirs(pathRdir);
        textS = fs.create(new Path(pathRdir.toString() + "/R"));
      }
      if (onlyflag != 2) {
        fs.mkdirs(pathSdir);
        textR = fs.create(new Path(pathSdir.toString() + "/S"));
      }
    }

    // Sample for S and R
    for (int k = 1; k <= nR; k++) {
      // generate afor random vector for XR
      for (int i = 0; i < dR; i++) {
        ri.xr[i] = myrand.nextGaussian();
      }

      si.forkey = k;
      for (int rr = 1; rr <= r; rr++) {
        // assign this point to a cluster
        si.label = (myrand.nextInt(2) % 2 == 0) ? -1 : 1;
        // generate a random vector for XS
        for (int i = 0; i < wlenr; i++) {
          si.xs[i] = myrand.nextGaussian();
        }
        // transform by w
        for (int i = 0; i < wlenr; i++) {
          si.xs[i] *= w[i];
        }
        sidkey.set(r * (k - 1) + rr); //si.sid = r * (k - 1) + rr;

        if (onlyflag != 1) {
          //output to S (SID Target ForeignKey XS)
          if (format == 0) {
            writerS.append(sidkey, si);
          } else {
            textS.writeBytes(si.writeString(sidkey.get()) + "\n");
          }
          //System.out.println("sid " + sidkey + " " + si.toString());
        }
      }

      for (int i = 0; i < dR; i++) {
        ri.xr[i] *= w[wlenr + i];
      }
      ridkey.set(k); //ri.rid = k;

      if (onlyflag != 2) {
        //output to R (RID XR)
        if (format == 0) {
          writerR.append(ridkey, ri);
        } else {
          textR.writeBytes(ri.writeString(ridkey.get()) + "\n");
        }
        //System.out.println("rid " + ridkey + " " + ri.toString());
      }

      // status
      if (k % 1000000 == 0) {
        System.out.println("Fin k = " + k);
      }
    }

    if (onlyflag != 1) {
      if (format == 0) {
        writerS.close();
      } else {
        textS.close();
      }
    }
    if (onlyflag != 2) {
      if (format == 0) {
        writerR.close();
      } else {
        textR.close();
      }
    }
  }
}
