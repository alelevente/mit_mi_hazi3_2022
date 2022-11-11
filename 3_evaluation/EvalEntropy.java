public class EvalEntropy{
	public static void main(String[] args){
		int above = Integer.parseInt(args[0]);
		int below = Integer.parseInt(args[1]);

		System.out.println(Solution.getEntropy(above, below));
	}
}
