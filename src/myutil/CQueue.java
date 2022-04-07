package myutil;

import java.util.LinkedList;

public class CQueue {  // 剑指 Offer 09. 用两个栈实现队列
    // 采用两个栈，一个栈用来append，一个栈用来delete
    // 如果delete stack无内容，则将append stack所有内容依次取出并放入delete并出队列

    /*  +++++++++++++++++++++++++++++++++++++++++++++++
    使用java的同学请注意，如果你使用Stack的方式来做这道题，会造成速度较慢；
    原因的话是Stack继承了Vector接口，而Vector底层是一个Object[]数组，那么就要考虑空间扩容和移位的问题了，同时比ArrayList多了多线程支持。
    可以使用LinkedList来做Stack的容器，因为LinkedList实现了Deque接口，所以Stack能做的事LinkedList都能做，其本身结构是个双向链表，扩容消耗少。
     */
    private LinkedList<Integer> stackAdd;
    private LinkedList<Integer> stackDel;

    public CQueue() {
        stackAdd = new LinkedList<Integer>();
        stackDel = new LinkedList<Integer>();
    }

    public void appendTail(int value) {
        stackAdd.addLast(value);
    }

    public int deleteHead() {  // 添加throws可能抛出的异常类型
        if (stackDel.isEmpty()) {
            if (stackAdd.isEmpty()) return -1;
            while (!stackAdd.isEmpty()) stackDel.addLast(stackAdd.removeLast());
        }
        return stackDel.removeLast();
    }

    public static void main(String[] args) {
        CQueue q = new CQueue();
        q.appendTail(5);
        int re1 = q.deleteHead();
        int re2 = q.deleteHead();
    }
}

