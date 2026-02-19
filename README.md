# NLP_HW2
#700773301_Leena Reddy Daida

PART1(Q5):
3. import numpy as np
cm = np.array([[5,10,5],
               [15,20,10],
               [0,15,10]])
precision=[]
recall=[]
for i in range(3):
    tp=cm[i,i]
    fp=sum(cm[i,:])-tp
    fn=sum(cm[:,i])-tp
    precision.append(tp/(tp+fp))
    recall.append(tp/(tp+fn))
print("Precision:",precision)
print("Recall:",recall)
macro_p=sum(precision)/3
macro_r=sum(recall)/3
micro= np.trace(cm)/np.sum(cm)
print("Macro Precision:",macro_p)
print("Macro Recall:",macro_r)
print("Micro Precision:",micro)
print("Micro Recall:",micro)

OUTPUT:                                                                                                                                              
Precision: [np.float64(0.25), np.float64(0.4444444444444444), np.float64(0.4)]
Recall: [np.float64(0.25), np.float64(0.4444444444444444), np.float64(0.4)]
Macro Precision: 0.36481481481481487
Macro Recall: 0.36481481481481487
Micro Precision: 0.3888888888888889
Micro Recall: 0.3888888888888889

II. PROGRAMMING
Q1: 
from collections import defaultdict
# 1. Training Corpus
corpus = [
    ["<s>","I", "love", "NLP", "</s>"],
    ["<s>","I", "love", "deep", "learning", "</s>"],
    ["<s>", "deep", "learning", "is", "fun", "</s>"]
]
# 2. Compute Unigram and Bigram Counts
unigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)

for sentence in corpus:
    for i in range(len(sentence)):
        unigram_counts[sentence[i]] += 1
        if i > 0:
            bigram_counts[(sentence[i-1], sentence[i])] += 1

print("\nUNIGRAM COUNTS:")
for word, count in unigram_counts.items():
    print(f"{word}: {count}")

print("\nBIGRAM COUNTS:")
for pair, count in bigram_counts.items():
    print(f"{pair}: {count}")
# 3. Estimate Bigram Probabilities (MLE)
# P(w2 | w1) = Count(w1,w2) / Count(w1)
def bigram_probability(w1, w2):
    if unigram_counts[w1] == 0:
        return 0
    return bigram_counts[(w1, w2)] / unigram_counts[w1]
# 4. Sentence Probability Function
def sentence_probability(sentence):
    prob = 1.0
    print("\nCalculating probability for:", " ".join(sentence))
    for i in range(1, len(sentence)):
        w1 = sentence[i-1]
        w2 = sentence[i]
        p = bigram_probability(w1, w2)
        print(f"P({w2}|{w1}) = {p}")
        prob *= p
    print("Total Probability =", prob)
    return prob
# 5. Test Sentences
s1 = ["<s>", "I", "love", "NLP", "</s>"]
s2 = ["<s>", "I", "love", "deep", "learning", "</s>"]
p1 = sentence_probability(s1)
p2 = sentence_probability(s2)
# 6. Model Preference
print("\nFINAL RESULT:")
if p1 > p2:
    print("Model prefers: <s> I love NLP </s>")
    print("Reason: Higher probability under bigram model (more frequent transitions)")
elif p2 > p1:
    print("Model prefers: <s> I love deep learning </s>")
    print("Reason: Higher probability under bigram model")
else:
    print("Both sentences equally probable")

Results:
UNIGRAM COUNTS:
<s>: 3
I: 2
love: 2
NLP: 1
</s>: 3
deep: 2
learning: 2
is: 1
fun: 1

BIGRAM COUNTS:
('<s>', 'I'): 2
('I', 'love'): 2
('love', 'NLP'): 1
('NLP', '</s>'): 1
('love', 'deep'): 1
('deep', 'learning'): 2
('learning', '</s>'): 1
('<s>', 'deep'): 1
('learning', 'is'): 1
('is', 'fun'): 1
('fun', '</s>'): 1

Calculating probability for: <s> I love NLP </s>
P(I|<s>) = 0.6666666666666666
P(love|I) = 1.0
P(NLP|love) = 0.5
P(</s>|NLP) = 1.0
Total Probability = 0.3333333333333333

Calculating probability for: <s> I love deep learning </s>
P(I|<s>) = 0.6666666666666666
P(love|I) = 1.0
P(deep|love) = 0.5
P(learning|deep) = 1.0
P(</s>|learning) = 0.5
Total Probability = 0.16666666666666666

FINAL RESULT:
Model prefers: <s> I love NLP </s>
Reason: Higher probability under bigram model (more frequent transitions)

