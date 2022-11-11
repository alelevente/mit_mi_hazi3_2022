public class EvalSeparation{
	public static void main(String[] args){
		int nRows = Integer.parseInt(args[0]);
		int nCols = Integer.parseInt(args[1]);
        
        int[][] features = new int[nRows][nCols];
        boolean[] labels = new boolean[nRows];
        
        //System.out.println("Features:");
        int arg_c=2;
        for (int i=0; i<nRows; i++){
            for (int j=0; j<nCols; j++){
                features[i][j] = Integer.parseInt(args[arg_c]);
                arg_c++;
                //System.out.println(features[i][j]);
            }
        }
        
        //System.out.println("Labels:");
        for (int i=0; i<nRows; i++){
            labels[i] = args[arg_c].equals("0")?false:true;
            arg_c++;
            //System.out.println(labels[i]);
        }

		int[] result = Solution.getBestSeparation(features, labels);
        System.out.println(result[0]+"@"+result[1]);
	}
}
