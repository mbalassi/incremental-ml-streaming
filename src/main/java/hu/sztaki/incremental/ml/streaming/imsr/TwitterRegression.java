/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package hu.sztaki.incremental.ml.streaming.imsr;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.flink.api.common.functions.GroupReduceFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.json.JSONParseFlatMap;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.apache.flink.util.Collector;
import org.apache.sling.commons.json.JSONArray;
import org.apache.sling.commons.json.JSONException;

public class TwitterRegression {

	public static int BatchSize = 10;

	// *************************************************************************
	// PROGRAM
	// *************************************************************************

	public static void main(String[] args) throws Exception {

		// set up the execution environment
		final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		env.setBufferTimeout(1);

		// get input data
		DataStream<String> stream = env.addSource(new TwitterSource("/home/tomi92/git/twitter.properties"),1);
		
		IMSR.MatrixSink sink = new IMSR.MatrixSink();
		
		stream
		.flatMap(new TwitterDoublePairFlatMap())
		.batch(BatchSize)
		.reduceGroup(new TwitterMatrixCreator())
		.map(new IMSR.MatrixMapper())
		.reduce(new IMSR.MatrixSumReducer())
		.addSink(sink);

		// execute program
		env.execute("Streaming Linear Regression (IMSR)");
	}

	public static class TwitterDoublePairFlatMap extends
	JSONParseFlatMap<String, Tuple2<Double, Double>> {
		private static final long serialVersionUID = 1L;
		
		/**
		 * Select the language from the incoming JSON text
		 */
		@Override
		public void flatMap(String value, Collector<Tuple2<Double, Double>> out) throws Exception {
			try{
				JSONArray ht = (JSONArray)get(value, "entities.hashtags");
				double x = ht.length();
				double y = (double)getString(value, "text").length();
				Tuple2<Double,Double> t = new Tuple2<Double, Double>(x, y);
				out.collect(t);
				//System.out.printf("%f %f\n", x , y);
			}
			catch (JSONException e){
				//System.err.print(e.getMessage());
			}
		}
	}

	
	public static class TwitterMatrixCreator
	implements GroupReduceFunction<Tuple2<Double, Double>, Tuple2<double[][], double[][]>>
	{
		private static final long serialVersionUID = 1143426179541008899L;
		
		@Override
		public void reduce(Iterable<Tuple2<Double, Double>> values,
				Collector<Tuple2<double[][], double[][]>> out) throws Exception {
			Array2DRowRealMatrix X = new Array2DRowRealMatrix(BatchSize, 1);
			Array2DRowRealMatrix y = new Array2DRowRealMatrix(BatchSize, 1);
			int i = 0;
			for(Tuple2<Double, Double> t : values)
			{
				if(i > BatchSize)
				{
					break;
				}
				X.setEntry(i, 0, t.f0);
				y.setEntry(i, 0, t.f1);
				++i;
			}
			out.collect(new Tuple2<double[][], double[][]>(X.getDataRef(),y.getDataRef()));
		}
		
	}
	
}
