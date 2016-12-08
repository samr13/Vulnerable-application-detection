import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
//import java.util.concurrent.TimeUnit;

public class CredentialsValidation {
	Map< String, String > credentials = new HashMap<String, String>();
	static int countUsername=0;
	static int countPassword=0;

	String curdir = System.getProperty("user.dir").replaceAll("\\\\", "/");
	
	public void connect() throws FileNotFoundException {
		String filepath = curdir+"/snapbuddyusers.csv";
		File f = new File(filepath);
		if(f.exists() && !f.isDirectory()) { 
			Scanner scanner = new Scanner(new File(filepath));
	        while(scanner.hasNextLine()){
	            String[] arr = scanner.nextLine().split(",");
	            credentials.put(arr[0], arr[1]);
	        }
	        scanner.close();
		} else {
			throw new FileNotFoundException("csv not found");
		}
	}
		
	public boolean validateUserName(String username)  {
		countUsername++;
		try {
			for ( String key : credentials.keySet() ) {
				if(username.equals(key))
					return true;
			}		
			Thread.sleep(1);  //user name doesn't match
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return false;
	}
	
	public boolean validatePassword(String username, String password) {
		
		countPassword++;
		try {
			//TimeUnit.MILLISECONDS.sleep(10);
			String pwd = credentials.get(username);
			for (int i=0; i < pwd.length() && i < password.length(); i++) {
				if (pwd.charAt(i)!=password.charAt(i)) {
					Thread.sleep(5); //password doesn't match
					return false;
				}
			}
			Thread.sleep(10);
			return true;
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return true;
	}
	
	public static void main(String[] args) throws FileNotFoundException {
		CredentialsValidation p = new CredentialsValidation();
		p.connect();

		//System.out.println(args[0]+","+args[1]);
		long start_time, end_time, exec_time;
		int id=0;
		String curdir = System.getProperty("user.dir").replaceAll("\\\\", "/");
		String filepath = curdir+"/mixeddataset.csv";
		FileWriter pw;
		Scanner scanner= new Scanner(new File(filepath));
	
		try {
			pw = new FileWriter("results.csv");
			pw.append("id, T1, f1, f2\n");
	        while(scanner.hasNextLine()){
	            String[] arr = scanner.nextLine().split(",");
	            CredentialsValidation.countUsername=0;
	            CredentialsValidation.countPassword=0;
	            start_time = System.nanoTime();
	    		if (p.validateUserName(arr[0])) {
	    			p.validatePassword(arr[0], arr[1]);
	    		} 
	    		end_time = System.nanoTime();
	    		exec_time = (end_time - start_time)/1000;
//	    		System.out.println(start_time+","+end_time+","+ exec_time);
	    		pw.append(id+","+exec_time+","+CredentialsValidation.countUsername+","+CredentialsValidation.countPassword+'\n');
//	    		pw.append(id+","+exec_time+","+p.countUsername+","+p.countPassword+'\n');
	    		id++;
	        }
	        scanner.close();
			pw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

}
