import java.util.Arrays;

public class Test {
    public void test(){
        int[] arr = new int[10];
        arr[0] = 0;
        arr[3] = 3;
        int[] arr1;
//        arr1 = arr[:];
        System.out.println(Arrays.toString(arr));
    }

    public static void main(String[] args) {
        Test fun = new Test();
        fun.test();
    }
}
