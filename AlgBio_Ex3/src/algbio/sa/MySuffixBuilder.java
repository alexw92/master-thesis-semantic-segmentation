package algbio.sa;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * 
 * @author Alexander Werthmann
 * 
 * TASK SHEET 3 - TASK 1 
 * Manber & Myers algorithm implementation
 * 
 * Use the main method of this class for getting the suffix arrays of hp1 data
 * 
 * Runtime comparison:
 * 
 * Calculating the suffix array of the hp_virus1 data on my running system the naive algorithm required 132.53
 * seconds of time while my implementation of the Manber&Myers algorithm run for 0.33 seconds.
 * 
 * Thus the faster algorithm saved 132 seconds and used only a fraction (0.249 %) of the naive algorithm. 
 * 
 */
public class MySuffixBuilder implements SuffixArrayBuilder {
	
	public static String fasta_hp1 = "fasta_hp1.txt";
	public static String fasta_hp2 = "fasta_hp2.txt";

	/**
	 * Implementation of Manber & Myers algorithm
	 */
	
	@Override
	public int[] build(String t) {
		return buildMM(t).stream().mapToInt(i -> i).toArray();
	}

	public ArrayList<Integer> buildMM(String t) {
		ArrayList<Integer> bucket = new ArrayList<Integer>();
		for(int i=0;i<t.length();i++)
			bucket.add(i);
		return bucketSort(t, bucket, 1);
	}
	
	protected ArrayList<Integer> bucketSort(String t, ArrayList<Integer> bucket, int step){		
		HashMap<String, ArrayList<Integer>> bmap = new HashMap<String, ArrayList<Integer>>();
		for(Integer i : bucket){
			String sub = t.substring(i,Math.min(i+step,t.length()));
			ArrayList<Integer> list = bmap.get(sub);
			if(list==null){
				list = new ArrayList<Integer>();
				bmap.put(sub, list);
			}
			list.add(i);				
		}
		ArrayList<Integer> res = new ArrayList<Integer>();
		ArrayList<String> sortedList = new ArrayList<String>();
		sortedList.addAll(bmap.keySet());
		Collections.sort(sortedList);
		for(String s : sortedList){
			ArrayList<Integer> val = bmap.get(s);
			if(val.size() > 1){
				res.addAll(bucketSort(t, val, step*2));
			}
			else{
				//Nothing left to sort, add remaining element to solution
				res.add(val.get(0));
			}
		}
		return res;
	}
	
	public static void main(String[] args){
		// read fasta file
		StringBuilder sb = new StringBuilder();
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fasta_hp1)))) {
			
			String line;
			while ((line=br.readLine())!=null) {
				if (line.startsWith(">")) {
					if (sb.length()>0) {
						System.err.println("WARNING: More than one sequence in fasta file, ignoring all but the first!");
						break;
					}
				}
				else
					sb.append(line);
			}			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		sb.append('$');
		NaiveSuffixArrayBuilder naiveBuilder = new NaiveSuffixArrayBuilder();
	//	String test = "MISSISSIPPI$";
		String test = sb.toString();
		MySuffixBuilder suffb = new MySuffixBuilder();
		
		long startMyers = System.currentTimeMillis();
		int[] res_myers = suffb.build(test);
		long endMyers = System.currentTimeMillis();
		float t1 = (endMyers-startMyers)/1000.0f ;
		System.out.printf("Time required for calculating suffix array using MM-algorithm: %.3f seconds",t1);
		
		System.out.println("Validation: "+naiveBuilder.check(test, res_myers)); 
		
		long startNaive = System.currentTimeMillis();
		int[] res_Naive= naiveBuilder.build(test);
		long endNaive = System.currentTimeMillis();
		float t2 = (endNaive-startNaive)/1000.0f;
		System.out.println("Time required for calculating suffix array using naive algorithm: "+t2+" seconds");
		
		System.out.println("Validation: "+naiveBuilder.check(test, res_Naive));
		System.out.printf("The MM-algorithm required only %.4f %% of the time needed by the naive algorithm.\n",(t1/t2)*100f);
	}


}
