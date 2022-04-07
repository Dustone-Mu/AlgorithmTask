package myutil;

import java.util.LinkedList;

// 剑指 Offer 30. 包含min函数的栈
class MinStack {
    // 采用LinkedList实现，采用"等长"的辅助栈保存当前最小值
    LinkedList<Integer> data = new LinkedList<>();
    LinkedList<Integer> min = new LinkedList<>();
    /** initialize your data structure here. */
    public MinStack() {
        min.add(Integer.MAX_VALUE);
    }

    public void push(int x) {
        data.add(x);
        min.add(Math.min(min.getLast(), x));
    }

    public void pop() {
        data.removeLast();
        min.removeLast();
    }

    public int top() {
        return data.getLast();
    }

    public int min() {
        return min.getLast();
    }
}