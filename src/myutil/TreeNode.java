package myutil;

import java.util.*;

public class TreeNode {
    public int val;
    public TreeNode left;
    public TreeNode right;

    public TreeNode() {}
    public TreeNode(int x) { val = x; }
    public TreeNode(TreeNode l, TreeNode r) { left = l; right = r; }
    public TreeNode(int x, TreeNode l, TreeNode r) { val = x; left = l; right = r; }

    public TreeNode(List<Integer> seq) {  // todo 序列化与反序列化 层序遍历转Tree
        /**
        * 将leetcode树相关的输入list转为Tree，和序列化有些差异，如果一个节点为null，则不表示其子节点。
        * root = new TreeNode(new ArrayList<>(Arrays.asList(5,3,6,2,4,null,8,1,null,7)));
        */
        int len = seq.size(), pos = 0;  // todo null会报错
        Queue<TreeNode> queue = new LinkedList<>();
        TreeNode root = new TreeNode(seq.get(pos++));
        queue.add(root);
        while (pos < len) {
            TreeNode node = queue.remove();
            if (node != null) {
                Integer val = seq.get(pos++);
                TreeNode child = val == null? null: new TreeNode(val);
                node.left = child;
                queue.add(child);
                if (pos >= len) break;
                val = seq.get(pos++);
                child = val == null? null: new TreeNode(val);
                node.right = child;
                queue.add(child);
            }
        }
        this.val = root.val;
        this.left = root.left;
        this.right = root.right;
    }

    public List<Integer> serialize() {
        /**
        * 树的序列化：层序遍历实现
        * @return List<Integer> 层序遍历.
        */
        List<Integer> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        TreeNode root = new TreeNode(this.val, this.left, this.right);
        queue.add(root);
        int pos = 0, len = 0;
        while (queue.size() != 0) {
            TreeNode node = queue.remove();
            res.add(node == null? null: node.val);
            pos++;
            if (node != null) {
                len = pos;
                queue.add(node.left);
                queue.add(node.right);
            }
        }
        return res.subList(0, len);
    }

    @Override
    public String toString() {
        return "TreeNode{" +
                "val=" + val +
                ", left=" + left +
                ", right=" + right +
                '}';
    }


    public static void main(){
        List<Integer> list = new ArrayList<>(Arrays.asList(5,3,6,2,4,null,8,1,null,7));
        //       5
        //    3      6
        //  2   4       8
        // 1   7
        TreeNode node = new TreeNode(list);
        System.out.print(node.toString() + '\n');
        List<Integer> relist = node.serialize();
        System.out.print(relist.toString());
    }
}


