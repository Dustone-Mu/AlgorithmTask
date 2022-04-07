package myutil;

import java.util.LinkedList;
import java.util.*;

// 剑指 Offer 59 - II. 队列的最大值
// 采用一个严格递减的队列保存最大值信息
public class MaxQueue {
    Queue<Integer> val;
    Deque<Integer> max;

    public MaxQueue() {
        val = new LinkedList<>();
        max = new LinkedList<>();
    }

    public int max_value() {
        if (max.isEmpty()) return -1;
        return max.peekFirst();
    }

    public void push_back(int value) {
        while (!max.isEmpty() && max.peekLast() < value) max.pollLast();
        max.offerLast(value);
        val.offer(value);
    }

    public int pop_front() {
        if (val.isEmpty()) return -1;
        int ans = val.poll();
        if (max.peekFirst() == ans) max.pollFirst();
        return ans;
    }
}

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue obj = new MaxQueue();
 * int param_1 = obj.max_value();
 * obj.push_back(value);
 * int param_3 = obj.pop_front();
 */


/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue obj = new MaxQueue();
 * int param_1 = obj.max_value();
 * obj.push_back(value);
 * int param_3 = obj.pop_front();
 */