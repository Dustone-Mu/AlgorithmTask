import java.util.ArrayList;
import java.util.List;
import java.util.ArrayList;
import java.util.List;
import myutil.*;


public class InterviewRecord {
    // 微软 22.3.13 笔试链接 2h 3tasks
    public int task1(String S) {  // todo rename
        // 微软笔试task1
        // write your code in Java SE 8
        // get length of each block and then calculate the res
        int N = S.length();
        if (N == 1) return 0;
        int blockLenMax = 0;

        // Step1: get the length of each block
        List<Integer> blockLen = new ArrayList<>();
        char pre = S.charAt(0);  // last char
        int start = 0;
        for (int i = 0; i < N; i++) {
            if (S.charAt(i) != pre) {
                pre = S.charAt(i);  // 当前block类型
                int len = i - start;
                blockLenMax = Math.max(blockLenMax, len);
                blockLen.add(len);  // 上一block长度
                start = i;
            }
        }
        blockLen.add(N - start);  // the last block
        blockLenMax = Math.max(blockLenMax, N - start);

        // Step2: calculate the res
        int res = 0;
        for (int len: blockLen) {
            res += (blockLenMax - len);
        }

        return res;
    }


    public int task2(int[][] A) {
        // write your code in Java SE 8
        // 如何转移到相近的位置？
        // 记录多余位置和空缺位置，进行一个笛卡尔积的路径距离计算，然后选一套路径组合.
        // 暴力可解，如何优化？

        return 0;
    }


    public int task3(int N) {
        // 微软笔试task3
        // range: 10^9 暴力不可取

        // Step1: int 2 int[]
        String N1str = Integer.toString(N + 1);
        int len = N1str.length() + 1;  // 需要多一位以供进位
        int[] nums = new int[len];
        for (int i = 1; i < len; i++) {
            nums[i] = N1str.charAt(i - 1) - 48;
        }

        // Step2: 首先从最高位开始检查，一旦和前一位相同，当前位++后续所有位置置0
        int pre = -1;
        for (int i = 0; i < len; i++) {
            if (nums[i] == pre) {
                nums[i]++;  // 自增
                carryFromPos(nums, i);  // 位置i的值+1,如果溢出则进位
                set0FromPos(nums, i);  // 后续位置置0
                i = -1;  // 当检测到相同临近值并进位之后必须重新开始检测
                pre = -1;
            } else {
                pre = nums[i];
            }
        }

        // Step3: int[] 2 int
        int res = 0;
        for (int num: nums) {
            res = res * 10 + num;
        }
        return res;
    }
    private void carryFromPos(int[] nums, int i) {
        while (i >= 0 && nums[i] > 9) {
            nums[i--] = 0;
            nums[i]++;
        }
    }
    private void set0FromPos(int[] nums, int i) {
        // i **之后**的位置置0
        while (++i < nums.length) {
            nums[i] = 0;
        }
    }


    public void MicrosoftTest() {
        InterviewRecord fun = new InterviewRecord();
        // 微软task1
        fun.task1("babaa");
    }


    // 阿里巴巴远程代码面试

//        public static void main(String[] args) {
//            InterviewRecord fun = new InterviewRecord();


}



