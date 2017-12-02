package algbio.sa;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * 
 * @author Alexander Werthmann
 * 
 *         TASK SHEET 3 - TASK 2 Longest common substring implementation
 *         
 *         The LCS of HP1 and HP2 has a length of 338 chars.
 * 		   Using the main-method of this class with reproduce the calculation
 * 
 *         What to proof:
 * 
 *         The following algorithm returns the longest common substring for two
 *         given strings s and t:
 * 
 *         1:  v = st, maxL = 0 
 *         2:  suffixArr = SuffixArray(v) 
 *         3:  for i in range(0,len(suffixArr)-1) 
 *         4:  		i1 = i 
 *         5:  		i2 = i+1 
 *         6: 		e1 = suffixArr[i1]
 *         7: 		e2 = suffixArr[i2] 
 *         8:  		if ( not(e1 in range(0, len(s)-1) and e2 in range(len(s),len(s)+len(t)-1) 
 *         9: 		AND not(e2 in range(0, len(s)-1) and e1 in range(len(s),len(s)+len(t)-1) ) 
 *         10: 			continue; 
 *         11: 		else 
 *         12: 			localMax= 0 
 *         13:			sub1 = v.subString(e1, end);
 *         13:			sub2 = v.substring(e2, end);
 *         13: 			for(j in range(0, min(len(sub1), len(sub2))-1) ) 
 *         14: 				if(sub1[j] == sub2[j]) 
 *         15: 					localMax++; 
 *         16: 				else 
 *         17: 					break; 
 *         18: 			if(localMax>maxL) 
 *         19: 				maxL = localMax
 * 
 * 
 *         The key idea to prove is that the longest common substring can be
 *         found in two consecutive elements (one suffix of s and one of t) of
 *         the suffix array of the concatenated string (so we do not have to
 *         compare n with n array elements). This can be proven easily: Let w =
 *         w1,w2,w3,...,wn be the longest common substring, which starts at
 *         index i in s and at index j in t. This implies that w is the prefix
 *         of both the suffixes i and i+j in v (remember v = st!) As the suffi
 *         xarray is sorted and w is the longest common substring of both of the
 *         strings s and t, there can't be another element between suffixes i
 *         and i+j in the suffix array.
 *         
 *         Runtime analysis:
 *         The preprocessing calculation of the suffix array by Manber&Myers is performed in O(nlog(n))
 *         The loop at line 3 runs n times and contains another loop at line 13 for comparing the string elements which 
 *         possibly could run for another n times. Although an heuristical runtime calculation for the average case would 
 *         return better values, the worst case runtime is O(n�)     
 *         
 */
public class LCS {

	public static String fasta_hp1 = "fasta_hp1.txt";
	public static String fasta_hp2 = "fasta_hp2.txt";
	private static int lenFirstString = 0;
	StringBuilder sb1;
	StringBuilder sb2;
	
	public static void main(String[] args){
		LCS lcs  = new LCS();
		lcs.calc_LCS();
	}

	public LCS() {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fasta_hp1)))) {
			sb1 = new StringBuilder();
			String line;
			while ((line = br.readLine()) != null) {
				sb1.append(line);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		lenFirstString = sb1.length();
		sb2 = new StringBuilder();
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fasta_hp2)))) {
			String line;
			while ((line = br.readLine()) != null) {
				sb2.append(line);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	/**
	 * Calculates the longest common substring of given strings a and b
	 * 
	 * @param a
	 * @param b
	 * @return the longest common substring of a and b
	 */
	public String calc_LCS() {		
		System.out.println("calculating LCS");
		String concat = sb1.toString() + sb2.toString(); 
		String res = null;
		MySuffixBuilder mysb = new MySuffixBuilder();
		int[] suffArr = mysb.build(concat);
		System.out.println("Suffix arrays constructed");
		int maxL = 0;
		String max= "";
		for (int i=0;i<suffArr.length-1;i++){
			int i1 = i;
			int i2 = i+1;
			int e1 = suffArr[i1];
			int e2 = suffArr[i2];
			if(i%20000==0)
				System.out.print(".");
			if(!(e1<lenFirstString && e2>=lenFirstString) && !(e2<lenFirstString && e1>=lenFirstString) )
				continue;
			else{
				int localMax = 0;
				String sub1, sub2;
				if(e1<lenFirstString){
					sub1 = concat.substring(e1,lenFirstString);
					sub2 = concat.substring(e2);
				}
				else{
					sub1 = concat.substring(e1);
					sub2 = concat.substring(e2,lenFirstString);					
				}
				for(int j=0;j<Math.min(sub1.length(), sub2.length());j++){
					if(sub1.charAt(j)==sub2.charAt(j))
						localMax++;
					else
						break;
					if(localMax > maxL){
						maxL = localMax;
						max = sub1.substring(0,localMax);
					}
				}
				
			}
		}
		System.out.println("\n LCS is of length "+maxL+":\n " + max);
		return max;
	}

}