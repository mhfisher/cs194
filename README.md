# cs194
Network theory research code on Preferential Attachment.

The paper motivating this research code be found [here](https://dl.acm.org/citation.cfm?id=3186122)

**Marshall's TODOs:**

1. Show that if some nodes play preferential attachment and other nodes play
different strategies (random selection, what else?), those nodes playing PA
have higher utilities than those who do not.
2. Show that longer games are more advantageous to PA strategy.
3. Big: Frame game in an optimization context and see if learned strategy is
similar to PA.
4. Weird but could be interesting: Optimize a strategy that is **not** allowed
to be PA (i.e. can not be an increasing function of node degree), and see
what this secondary strategy would be.

**Questions**

* What strategies are reasonable choices for a node other than PA?

1. Choose the most popular node.
2. Choose a node of average popularity (hedge your connection likelihood).
3. Use Tempered Preferential Attachment
4. Look at Lemma 4.15 for other suggestions.



