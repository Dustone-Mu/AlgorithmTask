import java.util.*;
import myutil.*;

public class CodingInterviewsOfferII {
    public int findRepeatNumber(int[] nums) {  // 剑指 Offer 03. 数组中重复的数字
        // 由于题目中规定了，数组范围为[0, n-1]，所以可以采用数组进行统计
        int[] count = new int[nums.length];
        for (int n: nums) {
            count[n]++;
            if (count[n] > 1) return n;
        } return -1;
    }

    public boolean findNumberIn2DArray(int[][] matrix, int target) {  // 剑指 Offer 04. 二维数组中的查找
        // 视为二叉搜索树，从右上角开始查看
        int m, n;
        m = matrix.length - 1;
        try {
            n = matrix[0].length - 1;
        } catch (Exception e) {
            return false;
        }
        if (m == -1 || n == -1) return false;  // 数组非空
        if (matrix[0][0] > target || matrix[m][n] < target) return false;  // 范围合理
        int i = 0, j = n;
        while (i <= m && j >= 0) {
            // 如果比节点要小，则必然在左边，如果比节点要大，则必然在下边
            if (target == matrix[i][j]) return true;
            else if (target < matrix[i][j]) j--;
            else i++;
        } return false;
//        // 通过大小对比限定出可能存在的范围
//        int m, n;
//        m = matrix.length - 1;
//        try {
//            n = matrix[0].length - 1;
//        } catch (Exception e) {
//            return false;
//        }
//        if (m == -1 || n == -1) return false;  // 数组非空
//        if (matrix[0][0] > target || matrix[m][n] < target) return false;  // 范围合理
//
//        int m0 = 0, n0 = 0;  // 确定四边界
//        while (target < matrix[m][0]) m--;
//        while (target < matrix[0][n]) n--;
//        while (target > matrix[m0][n] && m0 < m-1) m0++;
//        while (target > matrix[m][n0] && n0 < n-1) n0++;
//        for (int i=0; i<=m; i++) {
//            if (target == matrix[i][n] || target == matrix[i][n0]) return true;
//        }
//        for (int j=0; j<=n; j++) {
//            if (target == matrix[m][j] || target == matrix[m0][j]) return true;
//        }
//        return false;  // 不在四边界上
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {  // 剑指 Offer 07. 重建二叉树
        // 在中序遍历中，根节点分割两边，由于节点值不同，所以可以由pre确定根，由in确定左右
        // 由于array拷贝速度及内存都很慢，所以另写递归函数并调用
        int len = preorder.length;
        if (len == 0) return null;
        assert len == inorder.length;

        return buildTreebyIdx(preorder, inorder, 0, 0, len);  // 递归调用
    }
    private static TreeNode buildTreebyIdx(int[] preorder, int[] inorder, int pres, int ins, int len) {
        TreeNode root = new TreeNode(preorder[pres]);
        int i = 0;
        while (inorder[ins+i] != preorder[pres]) i++;  // 找到分割点
        if (i != 0) root.left = buildTreebyIdx(preorder, inorder, pres+1, ins, i);  // 存在左节点
        if (i != len-1) root.right = buildTreebyIdx(preorder, inorder, pres+i+1, ins+i+1, len-i-1);  // 存在右节点
        return root;
    }

    // 剑指 Offer 10- I. 斐波那契数列
    public int fib(int n) {
        // 采用一个状态转移的数学公式[1,1;1,0]^(n-1) .[0][0]elm
        // ----------------------------------
        // 采用循环，并使用一个长2的数组保存状态变量
        if (n < 2) return n;
        int[] nn = new int[] {0, 1};
        int id = 0;
        for (int i = 2; i <= n; i++) {
            id = i % 2;
            nn[id] = (nn[0] + nn[1]) % 1000000007;
        }
        return nn[id];
    }

    // 剑指 Offer 11. 旋转数组的最小数字
    public int minArray(int[] numbers) {
        // 重复，升序。二分查找解决排序数组问题，特例分析
        int end = numbers.length-1;
        if (end < 1) return numbers[0];
        int start = 0;
        int mid;
        while (end-start >= 2) {
            mid = (end + start)/2;
            if (numbers[mid] == numbers[start] && numbers[mid] == numbers[end]) {
                for (; start<end;) {
                    start++;
                    if (numbers[start] < numbers[start-1]) break;
                } return Math.min(numbers[0], numbers[start]);
            } else if (numbers[mid] >= numbers[start]) start = mid;
            else end = mid;
        }
        return Math.min(numbers[end], numbers[0]);  // 存在一些未进行旋转的例子
    }

    // 剑指 Offer 12. 矩阵中的路径
    public boolean exist(char[][] board, String word) {
        // 回溯法，深度优先遍历
        // 可统计一下board是否cover word对某些特殊情况可以加速
        int[][] sides = new int[][] {{0, 1, -1, 0}, {1, 0, 0, -1}};
        for (int i=0; i<board.length; i++)
            for (int j=0; j<board[0].length; j++)
                if (board[i][j] == word.charAt(0))
                    if (dfs(board, i, j, sides, word, 0)) return true;
        return false;
    }
    private static boolean dfs(char[][] board, int i, int j, int[][] sides, String word, int strPos) {
        // i, j表示当前字母的位置，strPos表示对应word里的位置
        if (strPos == word.length()-1) return true;
        board[i][j] = '0';  // 当前位置不可再选择，当失败时重置回去
        int inext, jnext;
        for (int k=0; k<4; k++) {  // 四个方向寻找
            inext = i + sides[0][k];
            jnext = j + sides[1][k];
            if (inext < 0 || inext >= board.length || jnext < 0 || jnext >= board[0].length)
                continue;
            if (board[inext][jnext] == word.charAt(strPos + 1))
                if (dfs(board, inext, jnext, sides, word, strPos+1))
                    return true;
        }
        board[i][j] = word.charAt(strPos);
        return false;
    }

    // 剑指 Offer 13. 机器人的运动范围
    public int movingCount(int m, int n, int k) {
        // 以10为长度，将矩阵分为大网格。每个大格子左上最小，右下最大，对角线相等。相邻大格子会整体差1,遵循网格内部的递增规则。
        // 1. 数学方法：寻找相关规律，进行数学计算，！存在障碍物！
        // 2. 回溯法+DFS：只往下往右就足够，进一步简化只在个位为0时往下
        boolean[][] grid = new boolean[m][n];
        return dfs13(grid, m, n, 0, 0, k);
    }
    private static int dfs13(boolean[][] grid, int m, int n, int i, int j, int k) {
        if (i >= m || j >= n || grid[i][j] || i/10 + i%10 + j%10 + j/10 > k) return 0;  // 如果访问过或者不可达或越界，返回0
        grid[i][j] = true;
        if (j%10 == 0) {// 只在个位为0时向下求索
            return 1 + dfs13(grid, m, n, i+1, j, k) + dfs13(grid, m, n, i, j+1, k);
        }
        return 1 + dfs13(grid, m, n, i, j+1, k);
    }

    // 剑指 Offer 14- I. 剪绳子
    public int cuttingRope(int n) {
        // 动态规划，从下而上的顺序
        // 采用一个中间数组保存这样的状态：对于位置i(i>3时)，记录最大切割值(因为此时切割乘必然大于原长)
        if (n == 2) return 1;
        if (n == 3) return 2;
        int[] cuts = new int[n+1];
        for (int i=0; i<4; i++) cuts[i] = i;
        for (int i=4; i<=n; i++) {
            for (int j=1; j<=i/2; j++) {
                cuts[i] = Math.max(cuts[i], cuts[j] * cuts[i-j]);
            }
        }
        return cuts[n];
        // 贪婪算法，尽量剪成3,其次是2的长度
    }
    public int cuttingRope0(int n) {
        // 贪婪算法，尽量剪成3,其次是2的长度。分割界为5,4
        if (n == 2) return 1;
        if (n == 3) return 2;

        int res = 1;
        while (true) {
            if (n >= 5) {
                res = res * 3;
                n -= 3;
            } else if (n == 4) {
                return res * 4;
            } else {
                return res * n;
            }
        }
    }

    // 剑指 Offer 15. 二进制中1的个数
    public int hammingWeight(int n) {
        // 位运算，通过将一个数和它减1之后的树进行与运算，可以将最右边一个1置0
        int counts = 0;
        while (n != 0) {
            n = n & (n-1);
            counts++;
        } return counts;
    }
    public int hammingWeight0(int n) {
        // 位运算，移动flag而不是源数字
        int flag = 1;
        int counts = 0;
        for (int i=0; i<32; i++) {
            if ((n&flag) != 0) counts++;
            flag = flag<<1;
        } return counts;
    }

    // 剑指 Offer 16. 数值的整数次方
    public double myPow(double x, int n) {
        // 位运算 递归优化 特例分析 异常检测
        // 无需考虑大数问题，但是有些计算结果会趋于0
        if (n == 0) return 1.0;
        if (Math.abs(x) == 1.0) return (n>>1 == 1 && x < 0)? -1.0: 1.0;
        if (x == 0.0 && n < 0) {
            System.out.println("n should bigger than 0 when x is equal to 0");
            return 0.0;
        }
//        double res = powUnsign(x, Math.abs(n), n<0);
        int absn = Math.abs(n);  // 此时可能会导致溢出
        double res = powUnsign(x, absn);
        if (n < 0 && res != 0.0) res = 1.0 / res;
        return res;
    }
    private double powUnsign(double x, int absn) {  // 可能会导致栈溢出
        if (absn > 1<<16) return 0.0;
        if (absn == 0) return 1;
        if (absn == 1) return x;
        double res = powUnsign(x, absn >> 1);
        res *= res;
        if ((absn & 1) == 1) res *= x;
        return res;
    }

    public double myPow0(double x, int n) {
        // x, n都有可能为负数，还需要考虑0
        // 优化：求幂 (x^n)^2 = x^2n
        if (n == 0) return 1.0;
        if (x == 0.0) return 0.0;

        int curn = 1;
        double res = x;
        while (2*curn <= n) {  // 先求合适的2^i次
            res = res * res;
            curn = curn * 2;
        }
        for (; curn < n; curn++) {
            res = res * x;
        }  // 由于精度损失，该结果可能为0

        return res;
    }

    // 剑指 Offer 17. 打印从1到最大的n位数
    public void printNumbers(int n) {
    }

    // 剑指 Offer 18. 删除链表的节点
    public ListNode deleteNode(ListNode head, int val) {
        if (head.val == val) return head.next;
        ListNode node = head;
        while (node.next.val != val) node = node.next;  // node.next 即为要删除的节点
        node.next = node.next.next;
        return head;
    }
    // 剑指 Offer 18. 删除链表的节点  原题-非阉割版
    public ListNode deleteNodeRaw(ListNode head, ListNode node) {  // todo 如何理解java中函数的对象引用与传递，引用的对比
        // O(1)，特例：删除head，tail，mid
        if (node == head) {
            head = head.next;
        } else if (node.next == null) {
            ListNode nodep = head;
            while (nodep.next != node) nodep = nodep.next;
            nodep.next = null;
        } else {
            node.val = node.next.val;
            node.next = node.next.next;
        }
        return head;
    }

    // 剑指 Offer 19. 正则表达式匹配
    public boolean isMatch(String s, String p) {
        // 97-122 a-z. The key is '*'. 特例 .*
        if (p.length() == 0 && s.length() != 0) return false;
        if (s.length() == 0) {
            int len = p.length();
            if ((len&1) != 0) return false;
            for (int i = 0; i<len; i++) {
                if (((i&1)==1? cins(p, i): '*') != '*') return false;
            } return true;
        }
        return matchStar(s, p, 0, 0);
    }
    private boolean matchStar(String s, String p, int sp, int pp) {
        if (sp == s.length() && pp == p.length()) return true;
        if (sp != s.length() && pp == p.length()) return false;
        // 如果存在*,则从0依次增加times，并递归调用查看是否匹配，函数应首先check当前是否有匹配要求，如果没有则按指针sp pp进行接下来的匹配
        if (cins(p, pp+1) == '*') {
            if (cins(s, sp) == cins(p, pp) || (cins(p, pp) == '.' && sp < s.length())) {
                return matchStar(s, p, sp+1, pp+2) ||
                        matchStar(s, p, sp+1, pp) ||
                        matchStar(s, p, sp, pp+2);
            } else {return matchStar(s, p, sp, pp+2); }  // match 0 times.
        }
        if (cins(s, sp) == cins(p, pp) || (cins(p, pp) == '.' && sp < s.length()))
            return matchStar(s, p, sp+1, pp+1);
        return false;
    }
    private char cins(String s, int pos) {
        return pos < s.length()? s.charAt(pos): '\n';
    }

    // 剑指 Offer 20. 表示数值的字符串
    public boolean isNumber(String s) {
        // 剔除开头的空格+第一个正负，，e/E +/- int
        int startN = 0;
        while (startN < s.length() && s.charAt(startN) == ' ') startN++;
        if (startN >= s.length()) return false;
        if (s.charAt(startN) == '+' || s.charAt(startN) == '-') startN++;  // 现在startN必为小数或者整数的开始
        int endE = s.length()-1;
        while (endE>=startN && s.charAt(endE)==' ') endE--;  // endE定位到最后一个非空格字符
        int endN = endE, startE = -1;  // startE同时作为是否存在e/E的flag
        for (int i = startN; i<=endE; i++) {  // 寻找e/E
            if (s.charAt(i) == 'e' || s.charAt(i) == 'E') {
                startE = i+1;
                if (startE > endE) return false;
                endN = i-1;
                break;
            }
        }
        // startN-endN必为小数或者整数
        if (startN > endN) return false;  // 幂必须存在
        int decimal = -1;
        for (int i=startN; i<=endN; i++) {
            if (s.charAt(i) == '.') {
                decimal = i;
                break;
            }
        }
        if (decimal != -1) {  // 存在小数部分
            if ((startN == decimal && endN == decimal) || !isInt(s, startN, decimal-1) || !isInt(s, decimal+1, endN)) return false;
        } else if (!isInt(s, startN, endN)) return false;
        if (startE != -1) {  // 存在指数部分,可为正负int
            if (s.charAt(startE) == '+' || s.charAt(startE) == '-') startE++;
            if (startE > endE || !isInt(s, startE, endE)) return false;
        }
        return true;
    }
    private boolean isInt(String s, int sp, int ep) {
//        if (sp>ep) return false;
        for (; sp<=ep; sp++) {
            if (s.charAt(sp) > '9' || s.charAt(sp) < '0') return false;
        } return true;
    }

    // 剑指 Offer 23. 链表中环的入口节点  leetcode没有
    public ListNode entranceOfListNode(ListNode head) { // todo leetcode没有
        // 采用快慢指针确定是否存在环
        ListNode pS = head, pQ = head;
        boolean existRing = false;
        while (pQ != null && pQ.next != null) {
            pS = pS.next;
            pQ = pQ.next.next;
            if (pQ == pS) {
                existRing = true;
                break;
            }
        }
        if (!existRing) return new ListNode(-1);  // 不存在环
        // 统计环的长度
        int lenRing = 1;
        pQ = pQ.next;
        while (pQ != pS) {
            lenRing++;
            pQ = pQ.next;
        }
        //剩余步骤和22一致
        return new ListNode(0);
    }

    // 剑指 Offer 24. 反转链表
    public ListNode reverseList(ListNode head) {
        if (head == null) return head;
        ListNode rehead = head, node = head.next, nextnode;
        rehead.next = null;
        while (node != null) {
            nextnode = node.next;
            node.next = rehead;
            rehead = node;
            node = nextnode;
        } return rehead;
    }

    // 剑指 Offer 25. 合并两个排序的链表
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // 新建ListNode
        ListNode reBeforeHead = new ListNode(0);
        ListNode node = reBeforeHead;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                node.next = l1;
                l1 = l1.next;
            } else {
                node.next = l2;
                l2 = l2.next;
            } node = node.next;
        }
        if (l1 == null) {
            node.next = l2;
        } else node.next = l1;
        return reBeforeHead.next;
    }

    // 剑指 Offer 26. 树的子结构
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        // 先在A中找到和B的跟节点相同的节点，之后再检查是否为包含关系
        if (A == null || B == null) return false;
        return findEqualRoot(A, B);
    }
    final static double epsilon = 0.000001;  // todo tips: 不可用==判断float是否相等
    private static boolean findEqualRoot(TreeNode A, TreeNode B) {
        if (A != null && B != null) {
            if (Math.abs(A.val - B.val) < epsilon) {
                if (containTree(A, B)) return true;
            }
            if (findEqualRoot(A.left, B)) return true;
            if (findEqualRoot(A.right, B)) return true;
        }
        return false;
    }
    private static boolean containTree(TreeNode A, TreeNode B) {
        // 如果B存在超出A的部分，则false
        if (B == null) return true;
        if (A == null || (Math.abs(A.val - B.val) > epsilon)) return false;
        return containTree(A.right, B.right) && containTree(A.left, B.left);
    }

    // 剑指 Offer 27. 二叉树的镜像
    public TreeNode mirrorTree(TreeNode root) {
        changeBranch(root);
        return root;
    }
    private static void changeBranch(TreeNode node) {
        if (node == null) return;
        TreeNode tempNode = node.right;
        node.right = node.left;
        node.left = tempNode;
        changeBranch(node.right);
        changeBranch(node.left);
    }

    // 剑指 Offer 28. 对称的二叉树
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return mirrorNode(root.left, root.right);
    }
    private static boolean mirrorNode(TreeNode node1, TreeNode node2) {
        if (node1 == null && node2 == null) return true;
        if (node1 == null || node2 == null || node1.val != node2.val) return false;
        return mirrorNode(node1.left, node2.right) && mirrorNode(node1.right, node2.left);
    }

    // 剑指 Offer 29. 顺时针打印矩阵
    public int[] spiralOrder(int[][] matrix) {
        int m = matrix.length;
        if (m == 0) return new int[0];
        int n = matrix[0].length;
        if (n == 0) return new int[0];
        int[] relist = new int[m * n];
        int pos = 0, round = 0;
        for (; (round+1)<<1 <= Math.min(m, n); round++) {
            for (int i = round; i < n - round; i++) {
                relist[pos] = matrix[round][i];
                pos++;
            } for (int i = round+1; i < m-round-1; i++) {
                relist[pos] = matrix[i][n-1-round];
                pos++;
            } for (int i = n-1-round; i >= round; i--) {
                relist[pos] = matrix[m-1-round][i];
                pos++;
            } for (int i = m-2-round; i > round; i--) {
                relist[pos] = matrix[i][round];
                pos++;
            }
        }
        if (m == 1 || n == 1) {
            for (int[] i: matrix) {
                for (int j: i) {
                    relist[pos] = j;
                    pos++;
                }
            }
        } else if ((m&1) == 1 && m <= n) {
            for (int i = round; i < n-round; i++) {
                relist[pos] = matrix[m>>1][i];
                pos++;
            }
        } else if ((n&1) == 1) {
            for (int i = round; i < m-round; i++) {
                relist[pos] = matrix[i][n>>1];
                pos++;
            }
        }
        return relist;
    }

    // 剑指 Offer 31. 栈的压入、弹出序列
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        // 如果pushed当前值和poped当前值相等，则pop，否则push
        int len = pushed.length;
        if (len != popped.length) return false;
        if (len == 0) return true;
        LinkedList<Integer> stack = new LinkedList<>();
        int i = 0, j = 0;
        while (i < len && j < len) {
            if (pushed[i] == popped[j]) {
                j++;
                while (stack.size() != 0 && stack.getLast() == popped[j]) {
                    stack.removeLast();
                    j++;
                }
            } else stack.add(pushed[i]);
            i++;
        }
        return (stack.size() == 0);
    }

    // 剑指 Offer 32 - I. 从上到下打印二叉树
    public int[] levelOrderI(TreeNode root) {
        // 层序遍历，采用Linkedlist存储需要打印的节点
        if (root == null) return new int[0];
        Queue<TreeNode> nodes = new LinkedList<TreeNode>();  // 不存储null节点
        nodes.add(root);
        List<Integer> reVal = new ArrayList<>();
        while (!nodes.isEmpty()) {
            TreeNode node = nodes.poll();
            reVal.add(node.val);
            if (node.left != null) nodes.add(node.left);
            if (node.right != null) nodes.add(node.right);
        }
        return reVal.stream().mapToInt(Integer::intValue).toArray();  // todo tips: LinkedList<Integer> 转 int[]
    }

    // 剑指 Offer 32 - II. 从上到下打印二叉树 II
    public List<List<Integer>> levelOrderII(TreeNode root) {
        // 用N记录节点数
        List<List<Integer>> reVal = new LinkedList<>();
        if (root == null) return reVal;
        Queue<TreeNode> nodes = new LinkedList<TreeNode>();
        nodes.add(root);
        int curN, nextN = 1;
        while (nextN != 0) {
            curN = nextN;
            nextN = 0;
            List<Integer> curVal = new LinkedList<>();
            while (curN != 0) {
                TreeNode node = nodes.poll();
                curVal.add(node.val);
                if (node.left != null) {nodes.add(node.left); nextN++;}
                if (node.right != null) {nodes.add(node.right); nextN++;}
                curN--;
            } reVal.add(curVal);
        } return reVal;
    }

    // 剑指 Offer 32 - III. 从上到下打印二叉树 III
    public List<List<Integer>> levelOrderIII(TreeNode root) {
        // 用reverse记录正逆向
        List<List<Integer>> reVal = new LinkedList<>();
        if (root == null) return reVal;
        LinkedList<TreeNode> nodes = new LinkedList<TreeNode>();
        nodes.add(root);
        int curN, nextN = 1;
        boolean reverse = false;
        while (nextN != 0) {
            curN = nextN;
            nextN = 0;
            List<Integer> curVal = new LinkedList<>();
            while (curN != 0) {
                TreeNode node = reverse? nodes.removeFirst(): nodes.removeLast();
                curVal.add(node.val);
                if (reverse) {
                    if (node.right != null) {nodes.addLast(node.right); nextN++;}
                    if (node.left != null) {nodes.addLast(node.left); nextN++;}
                } else {
                    if (node.left != null) {nodes.addFirst(node.left); nextN++;}
                    if (node.right != null) {nodes.addFirst(node.right); nextN++;}
                }
                curN--;
            } reVal.add(curVal);
            reverse = !reverse;
        } return reVal;
    }

    // 剑指 Offer 33. 二叉搜索树的后序遍历序列
    public boolean verifyPostorder(int[] postorder) {
        // 最后必为 root，可以划分左右分支
        int len = postorder.length;
        if (len <= 1) return true;
        int root = postorder[len-1];
        int rightStart = 0;
        while (postorder[rightStart] < root) rightStart++;  // 范围为[0, len-1]，左右分支可能不存在
        for (int i = rightStart; i < len; i++) {
            if (postorder[i] < root) return false;
        }
        return verifyPostorder(Arrays.copyOfRange(postorder, 0, rightStart)) && verifyPostorder(Arrays.copyOfRange(postorder, rightStart, len-1));
    }

    // 剑指 Offer 34. 二叉树中和为某一值的路径
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        List<List<Integer>> paths = new LinkedList<>();
        if (root == null) return paths;
        LinkedList<Integer> path = new LinkedList<>();
        pathSumLeft(paths, path, root, target);
        return paths;
    }
    private static void pathSumLeft(List<List<Integer>> paths, LinkedList<Integer> path, TreeNode node, int leftTarget) {
        // node != null, 递归调用，直到叶子节点查看是否equal，如果equal添加到paths中
        path.addLast(node.val);
        leftTarget -= node.val;
        if (node.left == null && node.right == null) { // 叶子节点
            if (leftTarget == 0) {
                paths.add(new LinkedList<>(path));
            }
            path.removeLast();
            return;
        }
        if (node.left != null) {
            pathSumLeft(paths, path, node.left, leftTarget);
        }
        if (node.right != null) {
            pathSumLeft(paths, path, node.right, leftTarget);
        }
        path.removeLast(); return;
    }

    // 剑指 Offer 35. 复杂链表的复制
    public Node copyRandomList(Node head) {
        // 原地拷贝node，之后做剖离并还原原node
        if (head == null) return null;
        Node node = head;
        while (node != null) {
            Node cpNode = new Node(node.val);
            cpNode.next = node.next;
            node.next = cpNode;
            node = cpNode.next;
        }
        node = head;
        while (node != null) {
            node.next.random = node.random == null? null: node.random.next;
            node = node.next.next;
        }
        node = head;
        Node cpHead = head.next, cpNode = head.next;
        while (node != null) {
            node.next = node.next.next;
            node = node.next;
            if (cpNode.next != null) {
                cpNode.next = cpNode.next.next;
                cpNode = cpNode.next;
            }
        } return cpHead;
    }

    // 剑指 Offer 36. 二叉搜索树与双向链表 - 结题 todo nice的解题方法
//    Node pre, head;
//    public Node treeToDoublyList(Node root) {
//        if(root == null) return null;
//        dfs(root);
//        head.left = pre;
//        pre.right = head;
//        return head;
//    }
//    void dfs(Node cur) {
//        if(cur == null) return;
//        dfs(cur.left);
//        if(pre != null) pre.right = cur;
//        else head = cur;
//        cur.left = pre;
//        pre = cur;
//        dfs(cur.right);
//    }
    // 剑指 Offer 36. 二叉搜索树与双向链表
    public TreeNode treeToDoublyList(TreeNode root) {  // 题目要求不能新建节点，但其实可以声明一个节点，但不 new
        if (root == null) return null;
        treeToList(root);
        while (root.left != null) root = root.left;  // 使用root索引List最左元素
        root.left = root;
        while (root.left.right != null) root.left = root.left.right;  // root.left指向最后一个元素
        root.left.right = root;
        return root;
    }
    private void treeToList(TreeNode node) {
        if (node.left != null) {
            treeToList(node.left);
            while (node.left.right != null) node.left = node.left.right;
            node.left.right = node;
        }
        if (node.right != null) {
            treeToList(node.right);
            while (node.right.left != null) node.right = node.right.left;
            node.right.left = node;
        }
    }

    // 剑指 Offer 38. 字符串的排列
    // dfs，排列组合类问题，使用数组存储元素，交换元素的顺序
    List<String> res = new LinkedList<>();
    char[] c;
    public String[] permutation(String s) {
        c = s.toCharArray();  // todo String.toCharArray  char[]
        dfs(0);
        return res.toArray(new String[res.size()]);  // todo List<String> 2 String[]
    }
    private void dfs(int layer) {
        if (layer == c.length-1) res.add(String.valueOf(c));  // todo char[] 2 String
        Set<Character> visited = new HashSet<>();
        for (int i=layer; i<c.length; i++) {
            if (visited.contains(c[i])) continue;
            visited.add(c[i]);
            swap(i, layer);
            dfs(layer+1);
            swap(i, layer);
        }
    }
    private void swap(int i, int j) {
        if (i == j) return;
        char temp = c[i];
        c[i] = c[j];
        c[j] = temp;
    }

    // 剑指 Offer 40. 最小的k个数
    // 哨兵+快排 速度最快 -----------------------------
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }
        // 最后一个参数表示我们要找的是下标为k-1的数
        return quickSearch(arr, 0, arr.length - 1, k - 1);
    }
    private int[] quickSearch(int[] nums, int lo, int hi, int k) {
        // 每快排切分1次，找到排序后下标为j的元素，如果j恰好等于k就返回j以及j左边所有的数；
        int j = partition(nums, lo, hi);
        if (j == k  || j == k+1) {
            return Arrays.copyOf(nums, j + 1);
        }
        // 否则根据下标j与k的大小关系来决定继续切分左段还是右段。
        return j > k? quickSearch(nums, lo, j - 1, k): quickSearch(nums, j + 1, hi, k);
    }
    // 快排切分，返回下标j，使得比nums[j]小的数都在j的左边，比nums[j]大的数都在j的右边。
    private int partition(int[] nums, int lo, int hi) {  // todo 快排
        int v = nums[lo];
        int i = lo, j = hi + 1;
        while (true) {
            while (++i <= hi && nums[i] < v);
            while (--j >= lo && nums[j] > v);
            if (i >= j) {
                break;
            }
            int t = nums[j];
            nums[j] = nums[i];
            nums[i] = t;
        }
        nums[lo] = nums[j];
        nums[j] = v;
        return j;
    }

    // 大根堆实现-------------------------------
    public int[] getLeastNumbersHeap(int[] arr, int k) {
        // 采用大根堆，堆的大小为k，不断poll出堆顶，剩下的就是最小的k个
        if (k == 0 || arr.length == 0) return new int[0];
        Queue<Integer> pq = new PriorityQueue<>((v1, v2) -> v2 - v1);  // todo 大根堆实现，PriorityQueue 重写比较器，方法：add/offer, element/peek, remove/poll
        for (int num: arr) {
            if (pq.size() < k) {
                pq.add(num);
            } else if (num < pq.peek()) {
                pq.poll();
                pq.add(num);
            }
        }
        int[] res = new int[pq.size()];
        int i = 0;
        for (int num: pq) {
            res[i++] = num;
        }
        return res;
    }

    // 以下为自己的解法
    public int[] mygetLeastNumbers(int[] arr, int k) {
        // 快速排序算法
        if (k == 0) return new int[0];
        partitionWithK(arr, 0, arr.length-1, k);
        return Arrays.copyOfRange(arr, 0, k);
    }
    void partitionWithK(int[] arr, int start, int end, int k) {
        if (start >= end) return;
        int mid = arr[end];
        int left = start, right = end-1;
        while (left < right) {
            while (arr[left] <= mid && left < right)
                left++;
            while (arr[right] >= mid && left < right)
                right--;
            swap40(arr, left, right);
        }
        if (arr[left] < mid) left++;
        swap40(arr, left ,end);
        if (left == k-1 || left == k) return;
        if (left > k) partitionWithK(arr, start, left-1, k);
        if (left < k-1) partitionWithK(arr, left+1, end, k);
    }
    void swap40(int[] arr, int i, int j) {
        if (i == j) return;
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    // 剑指 Offer 42. 连续子数组的最大和
    public int maxSubArray(int[] nums) {
        int res = 0, maxRes = Integer.MIN_VALUE;
        for (int num: nums) {
            res += num;
            maxRes = Math.max(maxRes, num);
            if (res < 0) res = 0;
            else maxRes = Math.max(res, maxRes);
        }
        return maxRes;
    }
    public int maxSubArrayDp(int[] nums) { // todo nice try，还有线段树解法需要再看
        // 动态规划解法，记录以第i个数字结尾的子数组的最大，之后返回最大的那个
        // 如果不修改原数组，需要O(n)空间
        if (nums.length == 0) return Integer.MIN_VALUE;
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            nums[i] += Math.max(nums[i-1], 0);
            res = Math.max(res, nums[i]);
        } return res;
    }

    // 剑指 Offer 44. 数字序列中某一位的数字
    public int findNthDigit(int n) {
        if (n < 10) return n;
        int i = 1, cur;
        while (true) {
            cur = (i == 1)? 10: i * 9 * (int) Math.pow(10, i-1);
            if (cur < 0 || n < cur) break;
            else {
                n -= cur;
                i++;
            }
        }
        // (int) Math.pow(10, i-1) + Math.floor(n/i)对应的是所在的数，i - n%i就是位数
        cur = (int) (Math.pow(10, i-1) + Math.floor(n/i));
        return (int) Math.floor(cur / (Math.pow(10, i-1-n%i))) % 10;
    }

    // 剑指 Offer 45. 把数组排成最小的数
    public String minNumber(int[] nums) {
        String[] numsS = new String[nums.length];
        for (int i = 0; i < nums.length; i++) numsS[i] = String.valueOf(nums[i]);
        // todo int[] to Integer[] https://www.cnblogs.com/cat520/p/10299879.html
        Arrays.sort(numsS, (a, b) -> (a + b).compareTo(b + a));
        return String.join("", numsS);
    }
//        Arrays.sort(numsI, new Comparator<Integer>(){  // todo 重写Arrays.sort比较器
//        public int compare(Integer a, Integer b) {
//            // 重写比较器，从首位数字开始对比，如果不等，则按首位升序。若相等，则去除首位继续对比
//            // 如果去除的首位就是最后一位，则另一个数的下个首位和上次的首位进行对比
//            if (a == b) return 0;
//            int i = (int) Math.floor(Math.log10(a)), j = (int) Math.floor(Math.log10(b));
//            int a1, b1 = 0, cur1;
//            boolean reverse = false;
//            while (i >= 0 && j >= 0) {
//                a1 = a / (int) Math.pow(10, i);
//                a = a % (int) Math.pow(10, i--);
//                b1 = b / (int) Math.pow(10, j);
//                b = b % (int) Math.pow(10, j--);
//                if (a1 != b1) {
//                    return a1 - b1;
//                } // else 进行下一级比较
//            }
//            // 此时必然有一方已耗尽首位数字，进入残局，假设剩余为a
//            if (j >= 0) {  // 若a已耗尽 reverse=true
//                reverse = true;
//                a = b; i = j;
//            }
//            while (i >= 0) {
//                a1 = a / (int) Math.pow(10, i);
//                a = a % (int) Math.pow(10, i--);
//                if (a1 != b1) {
//                    return  reverse? a1 - b1: b1 - a1 ;
//                }
//            } return 0;
//        }
//    });

    // 剑指 Offer 46. 把数字翻译成字符串
    public int translateNum(int num) {
        // 0-25: a-z，计算由数字构成的字符串有多少种翻译方法
        // 和跳格子类似，采用
        String numS = String.valueOf(num);
        int[] count = new int[] {1, 0};  // 分别表示最后一位为单个、最后一位与前面相连的翻译方法
        for (int i = 1; i < numS.length(); i++) {
            int countSingle = count[0];
            count[0] = count[0] + count[1];  // 当前位作为单个翻译
            if (numS.charAt(i-1) == '1' || (numS.charAt(i-1) == '2' && numS.charAt(i) <= '5')) count[1] = countSingle;
            else count[1] = 0;
        } return count[0] + count[1];
    }

    // 剑指 Offer 47. 礼物的最大价值
    public int maxValue(int[][] grid) {
        // 修改格子，改为到当前位置的最大奖励
        // 难点在于i != j时，
        int m = grid.length, n = grid[0].length;
        int i, j;
        for (int k = 0; k <= n + m; k++) {
            for (i = 0; i <= k; i++) {
                j = k - i;
                if (i >= m || j >= n) continue;
                grid[i][j] = grid[i][j] + Math.max((i-1)<0? 0: grid[i-1][j], (j-1)<0? 0: grid[i][j-1]);
            }
        } return grid[m-1][n-1];
    }

    // 剑指 Offer 48. 最长不含重复字符的子字符串
    public int lengthOfLongestSubstring(String s) {
        // 滑动窗口，每次扩大序列长度时检查是否当前扩大的字符是否已被包含，若是，子字符串起点设置为前一个之后
        // 动态规划，不断增加字符串长度，记录以当前位置为截止的最大长度
        int len = s.length();
        if (len == 0) return 0;
        int maxLen = 1;
        int start = 0, end = 0;
        Set<Character> set = new HashSet<>();
        set.add(s.charAt(0));
        while (start < len - maxLen) {
            while (end++ < len - 1) {
                char c = s.charAt(end);
                if (!set.contains(c)) {
                    set.add(c);
                    maxLen = Math.max(end - start + 1, maxLen);
                } else {
                    while (s.charAt(start) != c) {
                        set.remove(s.charAt(start++));
                    } start++;
                    break;
                }
            }
        }
        return maxLen;
    }
    // 剑指 Offer 48. 最长不含重复字符的子字符串 - 书上解法
    public int lengthOfLongestSubstring0(String s) {
        // 剑指offer解法：动态规划,其中数组的作用类似HashSet
        // f(i)表示到位置i的最长子字符串长度，采用一个26长数组记录字符出现的上一次位置，如果lastPos[s[i]]<maxLen则maxLen++
        int len = s.length();
        if (len == 0) return 0;
        int maxLen = 1, curlen = 1;
        for (int i = 1; i < len; i++) {
            // 终止，leetcode和原题不一致，并不要求字符全为小写字母
        } return 0;
    }

    // 剑指 Offer 49. 丑数
    public int nthUglyNumber(int n) {
        // 以空间换时间，采用数组保存从小到大的丑数，采用2,3,5作为base，并计算当前乘上的uglyNum，
        int[] top10 = new int[] {1, 2, 3, 4, 5, 6, 8, 9, 10, 12 };
        if (n <= 10) return top10[n-1];
        int[] ugly = new int[n];
        for (int i = 0; i < 10; i++) ugly[i] = top10[i];
        int[] base = new int[] {2, 3, 5};
        int[] basePos = new int[] {6, 4, 2};
        int[] baseNum = new int[] {16, 15, 15};
        int minBase = 1;
        for (int i = 10; i < n; i++) {
            ugly[i] = baseNum[minBase];
            baseNum[minBase] = ugly[++basePos[minBase]] * base[minBase];
            for (int j = 0; j < 3; j++) if (baseNum[j] < baseNum[minBase]) minBase = j;
        } return baseNum[minBase];
    }

    // 剑指 Offer 53 - II. 0～n-1中缺失的数字
    public int missingNumber(int[] nums) {
        // 脉冲函数，二分查找
        if (nums.length == 1) return 1 - nums[0];
        int start = 0, end = nums.length;
        if (nums[start] != 0) return 0; // 保证start为对应的，end为非对应的
        while (start < end - 1) {  // 最终目的，两者紧挨
            int mid = start + (end - start) / 2;
            if (nums[mid] == mid) start = mid;
            else end = mid;
        } return start == nums[start]? end: start;
    }

    // 剑指 Offer 54. 二叉搜索树的第k大节点
    public int kthLargest(TreeNode root, int k) {
        // 采用K个辅助空间进行右序遍历
        ArrayList<Integer> topK = new ArrayList<>(k);
        RoTwhitK(root, topK, k);
        return topK.get(k - 1);
    }
    private static void RoTwhitK(TreeNode node, ArrayList<Integer> topK, int k) { // right-order traversal
        if (node == null || topK.size() >= k) return;
        RoTwhitK(node.right, topK, k);
        if (topK.size() < k) {
            topK.add(node.val);
        } if (topK.size() < k) {
            RoTwhitK(node.left, topK, k);
        }
    }
//    int res, k;  // todo 利用类域优雅解题
//    public int kthLargest(TreeNode root, int k) {
//        this.k = k;
//        dfs(root);
//        return res;
//    }
//    void dfs(TreeNode root) {
//        if(root == null) return;
//        dfs(root.right);
//        if(k == 0) return;
//        if(--k == 0) res = root.val;
//        dfs(root.left);
//    }

    // 剑指 Offer 55 - I. 二叉树的深度
    int depthofTree = 0;
    public int maxDepth(TreeNode root) {
        depthofTree = 0;
        dfs(root, 0);
        return depthofTree;
    }
    void dfs(TreeNode node, int depth) {
        if (node == null) {
            depthofTree = Math.max(depth, depthofTree);
            return;
        }
        dfs(node.left, depth + 1);
        dfs(node.right, depth + 1);
    }

    // 剑指 Offer 55 - II. 平衡二叉树
    public boolean isBalanced(TreeNode root) {
        // 层序遍历？保存首个到达叶子节点的深度，以后每次到达叶子节点都和此值进行比较
        if (root == null) return true;
        int shallow = -1, depth = -1;
        Deque<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            depth++;
            int curNum = queue.size();
            while (curNum-- > 0) {
                TreeNode node = queue.poll();
                if (node == null) {
                    if (shallow == -1) shallow = depth; // 记录最浅的深度
                    else if (depth - shallow > 1) return false;
                } else {
                    queue.add(node.left);
                    queue.add(node.right);
                }
            }
        } return true;
    }

    // 剑指 Offer 56 - II. 数组中数字出现的次数 II
    public int singleNumber(int[] nums) {
        // todo 位运算 状态自动机 三进制
        int ones = 0, twos = 0;
        for(int num : nums){
            ones = ones ^ num & ~twos;
            twos = twos ^ num & ~ones;
        }
        return ones;
    }
    public int singleNumber0(int[] nums) {
        // 统计31位上1的个数，对3取余为1,则该位置为1
        int pos = 1 << 30;
        int count, res = 0;
        for (int i = 0; i < 31; i++) {
            count = 0;
            for (int num: nums) if ((num & pos) != 0) count++;
            res = res << 1;
            if (count % 3 == 1) res += 1;
            pos = pos >> 1;
        }
        return res;
    }

    // 剑指 Offer 60. n个骰子的点数
    public double[] dicesProbability(int n) {
        // 动态规划
        if (n == 1) return new double[] {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6};
        double[] pn1 = dicesProbability(n - 1);
        int len = pn1.length + 5;
        double[] res = new double[len];
        for (int i = 0; i < len; i++)
            res[i] = 1.0/6 * sumOfRange6(pn1, i);
        return res;
    }
    private double sumOfRange6(double[] nums, int end) {
        double res = 0;
        for (int i = Math.max(0, end - 5); i <= Math.min(nums.length - 1, end); i++) {
            res += nums[i];
        } return res;
    }



    // 剑指 Offer 62. 圆圈中最后剩下的数字
    public int lastRemaining(int n, int m) {  // todo 约瑟夫环
        // https://blog.csdn.net/u011500062/article/details/72855826
        int index = 0; // 当只有一人的时候 胜利者下标肯定为0
        for(int i = 2; i <= n; i++){
            index = (index + m) % i; // 每多一人 胜利者下标相当于往右挪动了m位,再对当前人数取模求得新的胜利者下标
        }return index;
    }

    // 剑指 Offer 63. 股票的最大利润
    public int maxProfit(int[] prices) {
        // 动态规划
        // space O(1), time O(n)
        int len = prices.length;
        if (len == 0) return 0;
        int maxP = 0;
        int minin = prices[0];
        for (int i = 1; i < len; i++) {
            if (minin > prices[i]) minin = prices[i];
            else maxP = Math.max(maxP, prices[i] - minin);
        } return maxP;
    }
    public int maxProfit0(int[] prices) {
        // 记录两个等长的int[]，一个记录到当前为止最低的价格，一个记录之后最高的价格，返回最大差价
        // space O(n), time O(n)
        int len = prices.length;
        if (len == 0) return 0;
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        int[] min_bf = new int[len];
        for (int i = 0; i < len; i++) {
            min = Math.min(min, prices[i]);
            min_bf[i] = min;
        }
        int maxP = 0;
        for (int i = len - 1; i >= 0; i--) {
            max = Math.max(max, prices[i]);
            maxP = Math.max(max - prices[i], maxP);
        } return maxP;
    }

    // 剑指 Offer 64. 求1+2+…+n
    public int sumNums(int n) {
        // todo 利用短路实现if的效果
        int sum = n;
        boolean flag = n > 0 && (sum += sumNums(n - 1)) > 0;
        return sum;
    }
    public int sumNums0(int n) {
        // 利用公式计算，使用了位运算以及幂运算
        // 此外，可以定义不等长的二维数组并直接返回二维数组的所有大小（C++中可以）
        return ((int) Math.pow(n, 2) + n) >> 1;
    }

    // 剑指 Offer 65. 不用加减乘除做加法
    public int add(int a, int b) {
        // todo 位运算，两数与可得进位的地方，两数或可得除了进位位置的加和，将这两位进行同样的相加操作
        if (a == 0) return b;
        int temp = b;
        b = a | b;
        a = a & temp;
        return add(a << 1, b & (~a));
    }

    // 剑指 Offer 66. 构建乘积数组
    public int[] constructArr(int[] a) {
        // 一个前向乘，一个后向乘
        if (a == null) return null;
        int len = a.length;
        if (len == 0) return new int[0];
        int[] bf = new int[len], af = new int[len];
        bf[0] = 1; af[len-1] = 1;
        for (int i = 0; i < len-1; i++) bf[i+1] = bf[i] * a[i];
        for (int i = len - 1; i > 0; i--) af[i-1] = af[i] * a[i];
        for (int i = 0; i < len; i++) af[i] = af[i] * bf[i];
        return af;
    }

    // 剑指 Offer 67. 把字符串转换成整数
    public int strToInt(String str) {
        int len = str.length();
        int i = 0;
        while (i < len && str.charAt(i) == ' ') i++;
        boolean negative = false;
        if (i < len && str.charAt(i) == '+') i++;
        else if (i < len && str.charAt(i) == '-') {
            i++; negative = true;
        }
        if (i >= len || !isNum(str.charAt(i))) return 0;
        int res = str.charAt(i) - '0';
        while (++i < len && isNum(str.charAt(i))) {
            long temp = (long) res * 10 + str.charAt(i) - '0';
            if (temp > Integer.MAX_VALUE) return negative? Integer.MIN_VALUE: Integer.MAX_VALUE;
            res = (int) temp;
        }
        return negative? -res: res;
    }
    private boolean isNum(char c) {
        return c >= '0' && c <= '9';
    }

    // 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {  // 优化写法
        if (p.val > root.val && q.val > root.val) return lowestCommonAncestor(root.right, p, q);
        else if (p.val < root.val && q.val < root.val) return lowestCommonAncestor(root.left, p, q);
        return root;
    }
    public TreeNode lowestCommonAncestor0(TreeNode root, TreeNode p, TreeNode q) {
        if (q.equals(root)) return q;
        else if (p.equals(root)) return p;
        if ((p.val > root.val ) ^ (q.val > root.val)) return root;
        if (p.val > root.val) return lowestCommonAncestor0(root.right, p, q);
        else return lowestCommonAncestor0(root.left, p, q);
    }

    // 剑指 Offer 68 - II. 二叉树的最近公共祖先
    public TreeNode lowestCommonAncestorE(TreeNode root, TreeNode p, TreeNode q) { // todo
        if(root == null) return null; // 如果树为空，直接返回null
        if(root == p || root == q) return root; // 如果 p和q中有等于 root的，那么它们的最近公共祖先即为root（一个节点也可以是它自己的祖先）
        TreeNode left = lowestCommonAncestorE(root.left, p, q); // 递归遍历左子树，只要在左子树中找到了p或q，则先找到谁就返回谁
        TreeNode right = lowestCommonAncestorE(root.right, p, q); // 递归遍历右子树，只要在右子树中找到了p或q，则先找到谁就返回谁
        if(left == null) return right; // 如果在左子树中 p和 q都找不到，则 p和 q一定都在右子树中，右子树中先遍历到的那个就是最近公共祖先（一个节点也可以是它自己的祖先）
        else if(right == null) return left; // 否则，如果 left不为空，在左子树中有找到节点（p或q），这时候要再判断一下右子树中的情况，如果在右子树中，p和q都找不到，则 p和q一定都在左子树中，左子树中先遍历到的那个就是最近公共祖先（一个节点也可以是它自己的祖先）
        else return root; //否则，当 left和 right均不为空时，说明 p、q节点分别在 root异侧, 最近公共祖先即为 root
    }
    public TreeNode lowestCommonAncestorII(TreeNode root, TreeNode p, TreeNode q) {
        LinkedList<TreeNode> pPath = new LinkedList<>();
        findPath(root, p, pPath);
        LinkedList<TreeNode> qPath = new LinkedList<>();
        findPath(root, q, qPath);
        TreeNode node = root;
        while (pPath.size() > 0 && qPath.size() > 0) {
            if (!pPath.element().equals(qPath.pollFirst())) break;
            node = pPath.pollFirst();
        }
        return node;
    }
    private boolean findPath(TreeNode root, TreeNode node, LinkedList<TreeNode> path) {
        if (root == null) return false;
        if (node.equals(root)) {
            path.addLast(node);
            return true;
        }
        if (findPath(root.right, node, path) || findPath(root.left, node, path)) {
            path.addFirst(root);
            return true;
        } return false;
    }


    // *剑指 Offer II *数据结构* 052. 展平二叉搜索树
    public TreeNode increasingBSTmid(TreeNode root) {
        // 最简单的方法，直接使用list进行中序遍历
        ArrayList<TreeNode> inorder = new ArrayList<> ();
        tree2Inorder(root, inorder);
        TreeNode pre = new TreeNode();
        for (TreeNode node: inorder) {
            pre.right = node;
            node.left = null;
            pre = node;
        } pre.right = null;
        return inorder.get(0);
    }
    private void tree2Inorder(TreeNode node, ArrayList<TreeNode> inorder) {
        if (node == null) return;
        tree2Inorder(node.left, inorder);
        inorder.add(node);
        tree2Inorder(node.right, inorder);
    }
    public TreeNode increasingBSTrec(TreeNode root) {  // todo 二叉搜索树转链表-递归法 不足：需要对先前的Tree遍历到最后一个节点
        // root 非null
        if (root == null) return null;
        TreeNode head = root;
        if (root.left != null) {
            head = increasingBSTrec(root.left);
            TreeNode node = head;
            while (node.right != null) node = node.right;
            node.right = root;
            root.left = null;
        }
        if (root.right != null) root.right = increasingBSTrec(root.right);
        return head;
    }
    private TreeNode tailNode52;
    public TreeNode increasingBST(TreeNode root) {  // todo 二叉搜索树转链表-递归法 优化，采用一个private节点保存之前flatten Tree的最后一个节点，即root
        TreeNode pre = new TreeNode();
        tailNode52 = pre;
        flattenTree(root);
        return pre.right;
    }
    private void flattenTree(TreeNode node) {
        if (node == null) return;
        flattenTree(node.left);
        tailNode52.right = node;
        tailNode52 = node;
        node.left = null;
        flattenTree(node.right);
    }



    public void test() {
        List<Integer> list = new LinkedList<>();
        list.add(10);
        List<Integer> list2 = new LinkedList<>(list);
        list2.add(30);
        System.out.println(list.toString());
        System.out.println(list2.toString());
    }
    public static void main(String[] args) {
        CodingInterviewsOfferII fun = new CodingInterviewsOfferII();
//        fun.test();

        // 52
        TreeNode root = fun.increasingBST(new TreeNode(new ArrayList<>(Arrays.asList(5,3,6,2,4,null,8,1,null,null,null,7,9))));
        System.out.print(root.serialize());
//        // 60
//        fun.dicesProbability(2);
//        // 56
//        fun.singleNumber(new int[] {3, 4, 3, 3});
//        // 67
//        fun.strToInt("-6147483648");
//        // 65
//        fun.add(111, 899);
//        // 63
//        fun.maxProfit(new int[] {7, 1, 5, 3, 6, 4});
//        // 48
//        fun.lengthOfLongestSubstring("abcabcbb");
//        // 45
//        fun.minNumber(new int[] {128, 12});
//        // 44
//        fun.findNthDigit(1000000000);
//        // 40
//        fun.getLeastNumbers(new int[] {0,0,1,2,4,2,2,3,1,4},8);
//        // 31
//        fun.validateStackSequences(new int[] {2,1,0}, new int[] {2,1,0});
//        // 29
//        fun.spiralOrder(new int[][] {{3},{2}});
//        // 20
//        fun.isNumber("0e");
//        // 19
//        fun.isMatch("", ".*");
//        // 16
//        fun.myPow(2.0, -2147483648);
//        // 13
//        fun.movingCount(1, 2, 1);
//        // 12
//        fun.exist(new char[][]{{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}}, new String("ABCCED"));
//        // 11
//        fun.minArray(new int[] {2,2,2,0,1});
//        // 07
//        TreeNode res7 = fun.buildTree(new int[] {3,9,20,15,7}, new int[] {9,3,15,20,7});
//        // 04
//        boolean res4 = fun.findNumberIn2DArray(new int[][] {{-5}}, -5);
//        // 03
//        int res3 = fun.findRepeatNumber(new int[] {2, 3, 1, 0, 2, 5, 3});
//        System.out.println(res3);

    }
}




