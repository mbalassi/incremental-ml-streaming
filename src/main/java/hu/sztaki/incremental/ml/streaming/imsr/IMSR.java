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

import org.apache.commons.math.linear.SingularValueDecompositionImpl;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.function.sink.SinkFunction;

public class IMSR {

	// *************************************************************************
	// PROGRAM
	// *************************************************************************

	public static void main(String[] args) throws Exception {

		// set up the execution environment
		final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

		// get arguments
		String fileName = "src/test/resources/1.csv";
		int batchSize = 5;
		if(args.length > 0)
		{
			fileName = args[0];
			if(args.length > 1)
			{
				batchSize = Integer.parseInt(args[1]);
			}
		}
		
		// get input data
		DataStream<Tuple2<double[][], double[][]>> stream = env.addSource(
				new MatrixVectorPairSource(fileName, batchSize), 1);
		
		MatrixSink sink = new MatrixSink();
		
		stream.map(new MatrixMapper())
		.reduce(new MatrixSumReducer())
		.addSink(sink);

		// execute program
		env.execute("Streaming Linear Regression (IMSR)");
	}

	// *************************************************************************
	// USER FUNCTIONS
	// *************************************************************************

	public static final class MatrixMapper
	implements MapFunction<Tuple2<double[][], double[][]>, Tuple2<double[][], double[][]>>
	{

		private static final long serialVersionUID = -5984071416255204043L;

		@Override
		public Tuple2<double[][], double[][]> map(Tuple2<double[][], double[][]> value)
				throws Exception {
			Array2DRowRealMatrix X = new Array2DRowRealMatrix( value.f0);
			Array2DRowRealMatrix y = new Array2DRowRealMatrix( value.f1);
			Array2DRowRealMatrix XT = new Array2DRowRealMatrix(X.transpose().getData());
			Array2DRowRealMatrix XTX = XT.multiply(X);
			Array2DRowRealMatrix XTy = XT.multiply(y);
			Tuple2<double[][], double[][]> res =
					new Tuple2<double[][], double[][]>(XTX.getDataRef(), XTy.getDataRef());
			return res;
		}
		
	}
	
	public static final class MatrixSumReducer
	implements ReduceFunction<Tuple2<double[][], double[][]>>
	{
		
		private static final long serialVersionUID = 1143426179541008899L;

		@Override
		public Tuple2<double[][], double[][]> reduce(Tuple2<double[][], double[][]> value1,
				Tuple2<double[][], double[][]> value2) throws Exception {
			Tuple2<double[][], double[][]> res = new Tuple2<double[][], double[][]>();
			Array2DRowRealMatrix M1 = new Array2DRowRealMatrix(value1.f0);
			Array2DRowRealMatrix M2 = new Array2DRowRealMatrix(value2.f0);
			Array2DRowRealMatrix v1 = new Array2DRowRealMatrix(value1.f1);
			Array2DRowRealMatrix v2 = new Array2DRowRealMatrix(value2.f1);
			res.f0 = M1.add(M2).getDataRef();
			res.f1 = v1.add(v2).getDataRef();
			return res;
		}
		
	}

	public static final class MatrixSink
	implements SinkFunction<Tuple2<double[][], double[][]>>
	{
		private static final long serialVersionUID = -7966965600616447076L;
		
		@Override
		public void invoke(Tuple2<double[][], double[][]> value) {
			Array2DRowRealMatrix M = new Array2DRowRealMatrix(value.f0);
			Array2DRowRealMatrix v = new Array2DRowRealMatrix(value.f1);
			Array2DRowRealMatrix invM = new Array2DRowRealMatrix(
					new SingularValueDecompositionImpl(M).getSolver().getInverse().getData());
			Array2DRowRealMatrix beta = invM.multiply(v);
			printVector(beta);
		}

		private void printVector(RealMatrix m)
		{
			assert(Math.min(m.getColumnDimension(), m.getRowDimension()) == 1);
			if(m.getColumnDimension() > 1)
			{
				m = m.transpose();
			}
			for(int i = 0; i<m.getRowDimension(); i++)
			{
				System.out.print(m.getEntry(i, 0));
				System.out.print(" ");
			}
			System.out.println();
		}
	}
	
}
