package algbio.sa;

import java.io.IOException;

/**
 * 
 * @author Alexander Werthmann
 * 
 * TASK SHEET 3 - TASK 1 
 * Manber & Myers algorithm implementation
 */
public class MySuffixBuilder implements SuffixArrayBuilder {
	
	public static String fasta_hp1 = "fasta_hp1";
	public static String fasta_hp2 = "fasta_hp2";

	/**
	 * Implementation of Manber & Myers algorithm
	 */
	@Override
	public int[] build(String t) {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	
	public static void main(String[] args){
		try {
			//Naive implementation Takes about 4 mins lol :D
			Main.main(new String[]{"fasta.txt"});
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		/*
		 * Test on small sample string
		 * */
		
		NaiveSuffixArrayBuilder naiveBuilder = new NaiveSuffixArrayBuilder();
		String test = "MISSISSIPI";
		int[] res_naive = naiveBuilder.build(test);

		for (int i :res_naive){
			System.out.println(i);
		}
		System.out.println(naiveBuilder.check(test, res_naive));
	}

}
