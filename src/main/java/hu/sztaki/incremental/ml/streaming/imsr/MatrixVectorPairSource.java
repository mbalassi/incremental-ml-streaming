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

import java.io.File;
import java.util.Scanner;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.function.source.SourceFunction;
import org.apache.flink.util.Collector;

public class MatrixVectorPairSource implements SourceFunction<Tuple2<double[][], double[][]>> {

	private static final long serialVersionUID = 2725224422071102334L;
	
	private String path;
	private int indepDim;
	private int windowSize;
	
	public MatrixVectorPairSource(String path, int indepDim, int windowSize) {
		this.path=path;
		this.indepDim=indepDim;
		this.windowSize=windowSize;
	}
	
	@Override
	public void invoke(Collector<Tuple2<double[][], double[][]>> out) throws Exception {
		File f = new File(path);
		if(!f.exists())
		{
			System.err.println(path + " does not exist.");
			System.exit(1);
		}
		Scanner s = new Scanner(f);
		while(s.hasNext())
		{
			Array2DRowRealMatrix X = new Array2DRowRealMatrix(windowSize, indepDim);
			Array2DRowRealMatrix y = new Array2DRowRealMatrix(windowSize, 1);
			readMatricesSideBySide(s, X, y);
			out.collect(new Tuple2<double[][], double[][]>(X.getDataRef(),y.getDataRef()));
		}
		s.close();
		out.close();
	}
	
	private void readMatricesSideBySide(Scanner scanner, RealMatrix... matrices)
	{
		for(int i = 0; i<matrices[0].getRowDimension(); i++)
		{
			if(!scanner.hasNextLine())
			{
				return; //there will be some 0 rows
			}
			String line = scanner.nextLine();
			Scanner lineScanner = new Scanner(line);
			for(RealMatrix m : matrices)
			{
				for(int j = 0; j<m.getColumnDimension(); j++)
				{
					double d = lineScanner.nextDouble();
					m.setEntry(i, j, d);
				}
			}
			lineScanner.close();
		}
	}
	
}
