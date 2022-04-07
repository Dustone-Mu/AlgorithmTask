import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Collectors;
import myutil.*;
import myutil.ListNode;

public class LeetcodeHot100 {

    // 15. 三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        // 先排序
        List<List<Integer>> res = new LinkedList<>();
        int len = nums.length;
        if (len == 0) return res;
        Arrays.sort(nums);
        int end;

        for (int i = 0; i < len - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            if (nums[i] > 0) break;
            end = len - 1;
            for (int j = i + 1; j < len -1; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                int left = -nums[i] - nums[j];
                if (left < 0) break;
                for (int k = end; k > j; k--) {
                    if (k < len -1 && nums[k] == nums[k + 1]) continue;
                    if (left == nums[k]) {
                        List<Integer> temp = Arrays.stream(new int[] {nums[i], nums[j], nums[k]}).boxed().collect(Collectors.toList());
                        // if (!res.isEmpty() && res.get(res.size() - 1).equals(temp)) continue;
                        res.add(temp);
                    } else if (left > nums[k]) {
                        break;
                    } else {
                        end--;
                    }
                }
            }
        }
        return res;
    }


    // 22. 括号生成
    char[] path;
    List<String> res;
    public List<String> generateParenthesis(int n) {
        // 堆栈实现？排序问题？嵌套结构如何表现？由内到外：从()开始拓展？树好表现吗？遍历？填空游戏？
        // 使用一个计数代替栈，表示当前左括号的数量
        res = new ArrayList<>();
        path = new char[2 * n];
        build(0, 0, n, n);
        return res;
    }
    private void build(int pos, int ns, int nl, int nr) {
        if (nr == 0) {
            res.add(String.valueOf(path));
            return;
        }
        if (ns != 0) {  // 不为0时才能用 ）
            path[pos] = ')';
            build(pos + 1, ns - 1, nl, nr - 1);
        }
        if (nl != 0) {
            path[pos] = '(';
            build(pos + 1, ns + 1, nl - 1, nr);
        }
    }


    // 142. 环形链表 II  22.2.1 17:24
    public ListNode detectCycle(ListNode head) {  // todo 快慢指针 Floyd判圈算法
        // 快慢指针，经计算分析，当快慢指针相遇之后，采用另一指针指向开头，并和slow一起移动，当再次相遇时，即为入点
        ListNode l = head, f = head, res = head;
        while (f != null) {
            l = l.next;
            if (f.next != null) {
                f = f.next.next;
            } else {
                return null;
            }
            if (f == l) {
                while (res != l) {
                    res = res.next;
                    l = l.next;
                }
                return res;
            }
        }
        return null;
    }

    public ListNode detectCycle_raw(ListNode head) {
        // 双指针，快慢指针  采用cnt记录两次相遇之间的距离，之后采用距离为cnt的快慢指针来检测入点
        if (head == null) return null;
        ListNode l = head, f = head;
        boolean exist = false;
        int cnt = 0;
        while (f.next != null && f.next.next != null) {
            l = l.next;
            f = f.next.next;
            if (exist) cnt++;
            if (l == f) {
                if (exist) break;
                else exist = true;
            }
        }
        if (!exist) return null;
        l = head; f = head;
        while (cnt-- != 0) f = f.next;
        while (l != f) {
            l = l.next;
            f = f.next;
        }
        return l;
    }


    // 148. 排序链表
    public ListNode sortList(ListNode head) {
        // 链表归并排序+快慢指针：针对链表排序最好的方法
        // 将大于两个节点的链按快慢指针(遍历两次其实是更为简单的)分为两部分，两部分分别递归排序，排序好之后组合返回
        if (head == null || head.next == null) return head;  // 至少两个节点才值得被合并
        ListNode fast = head, slow = head, temp = head;
        while (fast != null && fast.next != null) {  // todo code 链表快慢指针二分法
            fast = fast.next.next;
            temp = slow;
            slow = slow.next;
        }  // slow 对应位置即为后半段，长为奇时比前半段多一个节点
        temp.next = null;
        fast = head;
        fast = sortList(fast);
        slow = sortList(slow);
        head = new ListNode(-1);
        temp = head;
        while (fast != null && slow != null) {
            if (fast.val > slow.val) {
                temp.next = slow;
                slow = slow.next;
            } else {
                temp.next = fast;
                fast = fast.next;
            }
            temp = temp.next;
        }
        temp.next = fast == null? slow: fast;
        return head.next;
    }
    public ListNode sortList0(ListNode head) {
        // 暴力 插入排序
        ListNode prehead  = new ListNode(Integer.MIN_VALUE);
        ListNode renode, temp;
        while (head != null) {
            renode = prehead;  // 指向新进节点的前一节点
            while (renode.next != null && renode.next.val < head.val) {
                renode = renode.next;
            }
            temp = head.next;
            head.next = renode.next;
            renode.next = head;
            head = temp;
        }
        return prehead.next;
    }


    // 448. 找到所有数组中消失的数字
    public List<Integer> findDisappearedNumbers(int[] nums) {
        // [优秀解法 骚操作 hashmap 取余] 重点在于 todo how 如何让某个位置的值同时表示该位置的状态和原始数据
        // 官解：由于 \textit{nums}nums 的数字范围均在 [1,n][1,n] 中，我们可以利用这一范围之外的数字，来表达「是否存在」的含义。(也可采用取反)
        for (int num: nums) {
            nums[Math.abs(num) - 1] = -Math.abs(nums[Math.abs(num) - 1]);
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) res.add(i + 1);
        }
        return res;
    }
    public List<Integer> findDisappearedNumbers_My(int[] nums) {
        // [自写解法较为复杂]，将数放到应在的范围：如果一个位置不匹配，则和值对应位置进行互换，直到匹配或者互换的位置上自匹配
        for (int i = 0, change; i < nums.length; i++) {
            while (nums[i] != i + 1) {
                change = nums[i];
                if (nums[change - 1] == change) break;
                nums[i] = nums[change - 1];
                nums[change - 1] = change;
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) res.add(i + 1);
        }
        return res;
    }


    // 300. 最长递增子序列
    public int lengthOfLIS(int[] nums) {
        // dp: 使用maxi记录使用位置i为最后一位的子序列的最大长度，
        // 转移：maxi[i+1] 使用之前所有ress进行匹配
        if (nums.length == 1) return 1;
        int[] maxi = new int[nums.length];
        maxi[0] = 1;
        int res = 1;
        for (int i = 1; i < nums.length; i++) {
            maxi[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i])
                    maxi[i] = Math.max(maxi[i], maxi[j] + 1);
            }
            res = Math.max(res, maxi[i]);
        }
        return res;
    }
    public int lengthOfLIS0(int[] nums) {
        // 贪心+二分查找，有点意思的解乏
        // todo
        return 0;
    }


    // 322. 零钱兑换
    public int coinChangeUp(int[] coins, int amount) {  // todo 动态规划 自下而上 better
        int[] count = new int[amount+1];
        Arrays.fill(count, amount + 1);
        count[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin: coins) {
                if (coin <= i) {
                    count[i] = Math.min(count[i], count[i - coin] + 1);
                }
            }
        }
        return count[amount] > amount? -1: count[amount];
    }

    public int coinChange(int[] coins, int amount) {  // todo 动态规划 自上而下
        // 采用动态规划求解，F(S)为S金额下的最小个数，则F(S)=min 1+F(S-ci)，对coins进行遍历
        // 采用hashmap来保存中间状态，采用-1表示不可凑出
        coins = Arrays.stream(coins).boxed().sorted((a, b) -> b - a).mapToInt(p -> p).toArray();  // todo int[]排序
        HashMap<Integer, Integer> FS = new HashMap<>();
        FS.put(0, 0);
        return helper322dp(coins, amount, FS);
    }
    private int helper322dp(int[] coins, int amount, HashMap FS) {
        if (amount < 0) return -1;
        if (FS.containsKey(amount)) return (int) FS.get(amount);
        int num = Integer.MAX_VALUE;
        for (int coin: coins) {
            int numi = helper322dp(coins, amount - coin, FS);
            if (numi != -1) num = Math.min(num, numi);
        }
        FS.put(amount, num == Integer.MAX_VALUE? -1: num+1);
        return (int) FS.get(amount);
    }

    public int coinChangeArray(int[] coins, int amount) {  // todo 动态规划 数组替代hashmap 速度更快
        if (amount < 1) {
            return 0;
        }
        return coinChangeArray(coins, amount, new int[amount]);
    }
    private int coinChangeArray(int[] coins, int rem, int[] count) {
        if (rem < 0) {
            return -1;
        }
        if (rem == 0) {
            return 0;
        }
        if (count[rem - 1] != 0) {
            return count[rem - 1];
        }
        int min = Integer.MAX_VALUE;
        for (int coin : coins) {
            int res = coinChangeArray(coins, rem - coin, count);
            if (res >= 0 && res < min) {
                min = 1 + res;
            }
        }
        count[rem - 1] = (min == Integer.MAX_VALUE) ? -1 : min;
        return count[rem - 1];
    }


    // 287. 寻找重复数  22.3.31
    public int findDuplicate(int[] nums) {  // todo 快慢指针 数组-链表转换 Floyd判圈算法 141 142
        // 采用快慢指针方法，将数组视为链表，相同的数即代表同一节点，代表环的入口  O(n) THE BEST
        int n = nums.length;  // nums值均为 1~n-1
        int l = 0, f = 0, res = 0;
        do {
            l = nums[l];
            f = nums[nums[f]];
        } while (l != f);
        while (l != res) {
            l = nums[l];
            res = nums[res];
        }
        return res;
    }

    public int findDuplicate_Bit(int[] nums) {  // todo 二进制 位运算
        // 采用二进制位运算，计算第i位的1的个数，若nums的和比1~n的和多，则该位置为1  O(nlogn)
        int pos = 1, res = 0, cnts, cntn;
        int len = nums.length;
        while (true) {
            cnts = 0;
            cntn = 0;
            for (int i = 0; i < len; i++) {
                cnts += (pos & nums[i]) == 0? 0: 1;
                cntn += (pos & i) == 0? 0: 1;
            }
            if (cntn == 0) break;
            res = res | (cnts > cntn? pos: 0);
            pos = pos << 1;
        }
        return res;
    }

    public int findDuplicate_BinS(int[] nums) {  // todo 二分查找 边界条件
        // 二分查找，计算小于i的数的个数  O(nlogn)
        int n = nums.length;
        int l = 1, r = n - 1, res = -1;
        while (l <= r) {
            int mid = (l + r) >> 1;
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (nums[i] <= mid) count++;
            }
            if (count <= mid) {
                l = mid + 1;  // 通过范围偏移来打破while循环
            } else {
                r = mid - 1;
                res = mid;
            }
        }
        return res;
    }


    // 309. 最佳买卖股票时机含冷冻期 22.3.31
    public int maxProfit(int[] prices) {
        // 动态规划：
        // 当天三种状态(持有，未持有，冷冻期)下的收益情况
        // 遍历，状态转移图
        int p1 = 0;
        int p0 = 0;
        int pf = 0;
        int a, b, c;
        for (int i = 1; i < prices.length; i++) {
            a = Math.max(Math.max(p0, pf), p1 + prices[i] - prices[i-1]);
            b = Math.max(pf, p0);
            c = p1;
            p1 = a; p0 = b; pf = c;
        }
        return Math.max(pf, Math.max(p0, p1));
    }


    public void quickSort(int[] a, int l, int r){  // todo 快速排序
        if (l < r){
            int temp = a[l];
            while (l < r){
                while (l < r && a[r] > temp){
                    r--;
                }
                if (l < r){
                    a[l++] = a[r];
                }
                while (l < r && a[l] <= temp){
                    l++;
                }
                if (l < r){
                    a[r--] = a[l];
                }
            }
            a[l] = temp;
            quickSort(a, l, temp-1);
            quickSort(a, temp+1, r);
        }
    }


    // 347. 前 K 个高频元素
    public int[] topKFrequent(int[] nums, int k) {  // todo 快速排序求 topK
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num: nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        List<Integer> values = new ArrayList<>(map.values());
        int kth = qSort347(values, k);
        return null;
    }
    private int qSort347(List<Integer> values, int k) {
        return -1;
    }

    public int[] topKFrequent_queue(int[] nums, int k) {  // todo 小跟堆求 topK
        /* hashmap存储 值-个数 的映射，*/
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num: nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        for (int cnt: map.values()) {
            if (queue.size() < k) {
                queue.add(cnt);
            } else {
                if (cnt > queue.peek()) {
                    queue.remove();
                    queue.add(cnt);
                }
            }
        }
        List<Integer> res = new ArrayList<>();
        int kth = queue.peek();
        for (int num: map.keySet()) {
            if (map.get(num) >= kth) res.add(num);
        }
        return res.stream().mapToInt(Integer::intValue).toArray();
    }



    public static void main(String[] args) {
        LeetcodeHot100 fun = new LeetcodeHot100();

        // 347. 前 K 个高频元素
        fun.topKFrequent(new int[] {1,1,1,2,2,3}, 2);
        // 309. 最佳买卖股票时机含冷冻期 22.3.31
        fun.maxProfit(new int[] {1,2,3,0,2});
        // 322. 零钱兑换
        fun.coinChange(new int[] {1,2,5}, 11);
        // 300. 最长递增子序列
        fun.lengthOfLIS(new int[] {10,9,2,5,3,7,101,18});
        // 22. 括号生成
        fun.generateParenthesis(3);
        // 15. 三数之和
        fun.threeSum(new int[] {0,0,0,0});
    }
}
