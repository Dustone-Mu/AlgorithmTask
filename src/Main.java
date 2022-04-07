import java.util.Scanner;

/*
ACM 模式，自己处理输入输出
 */
public class Main {
    public static void minLenArray(int N, int L) {
        // 连续非负整数，长>=L,sum=N
        // 要找出长度最小的，可以从后往前找，
        // 动态规划，如果当前和大于N，r左移，如果当前和小于N，l左移，移动过程中注意长度与边界限制
        int l = N - L, r = N - 1, len = L;
        int sum = (int) (l + r) * L / 2;
        while (l >= 0) {
            if (sum == N) break;
            if (sum < N || len <= L) {  // 当sum不足，或者len不能再减少时，左移l
                l--;
                len++;
                sum += l;
            } else {
                sum -= r;
                r--;
                len--;
            }
        }
        if (l < 0 || sum != N) {
            System.out.println("No");
        } else {
            StringBuilder sb = new StringBuilder();
            sb.append(l);
            for (int i = l+1; i <= r; i++) sb.append(" "+i);
            System.out.println(sb.toString());
        }
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int N = in.nextInt();
        int L = in.nextInt();
        minLenArray(N, L);
    }
}