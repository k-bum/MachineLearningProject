# MachineLearningProject
Stock price forecast using Transformer

# Transformer

- Transformer : Attention is all you need

<기존 Seq2Seq 모델의 한계점>

- 고정된 크기의 context vector에 문장의 정보를 압축하는 과정에서 병목(bottle neck)이 발생해 성능 하락의 원인이 된다
- 새로운 단어가 입력될 때마다 hidden state 값을 갱신 → 이전의 정보를 모두 포함하고 있다 → 마지막 hidden state 값을 모든 단어의 정보를 포함하고 있는 context vector로 설정
- 문장의 길이가 긴 경우 처리하는 것이 어렵다(앞에 위치한 단어의 정보가 encoding 과정에서 손실된다)
- 디코더가 context vector를 매번 참고하면 성능을 기존보다 향상할 수는 있으나 context vector가 모든 문장의 정보를 가지고 있어야 하므로 성능이 저하

→ 매번 인코더의 모든 출력(hidden state)을 디코더에서 참고한다(attention mechanism)

디코더는 매번 인코더의 모든 출력 중에서 어떤 정보가 중요한지를 계산(energy) → energy 값에 softmax를 적용해 가중치 값을 인코더의 hidden state 값에 적용해 사용

energy는 인코더의 모든 출력값들 중에서 어떤 값과 연관성이 가장 높은지 계산한 값이다 → attention 가중치를 사용해 각 출력이 어떤 입력 정보를 참고했는지 알 수 있다

transformer는 RNN, CNN을 전혀 사용하지 않는다(인코더와 디코더를 여러 개 사용) → 문장 내 각 단어들의 순서에 대한 정보를 주기 위해서 Positional encoding(위치 정보를 포함하고 있는 embedding)을 사용

Positional encoding : transformer는 단어 입력을 순차적으로 받는 방식이 아니기 때문에 단어의 위치 정보를 알려줘야 하므로 각 단어의 embedding vector에 위치 정보를 더해 모델의 입력으로 사용한다 → 같은 단어라고 하더라도 문장 내의 위치에 따라 입력으로 들어가는 embedding vector의 값이 달라진다

<img width="432" alt="a" src="https://github.com/k-bum/MachineLearningProject/assets/96854885/2f71c80d-b28c-4116-aff4-ae6bbd44e3b4">

임베딩이 끝난 이후에 attention을 진행

![Untitled](https://github.com/k-bum/MachineLearningProject/assets/96854885/c0091316-3b6d-4b7e-b44b-afc1f6ffa94a)

attentetion

1. **Self attention** : attention을 자신에게 수행하는 경우(입력된 문장 내 단어 간의 관계)
    
    → encoder self attention, masked decoder self attention : 매번 입력 문장에서 각 단어가 어떤 다른 단어와 연관성이 높은지 계산
    
    인코더의 셀프 어텐션의 경우 모든 단어에 대해 계산이 이뤄지지만, 디코더의 셀프 어텐션의 경우 현재 시점의 예측 단어에서 현재 시점보다 미래에 있는 단어들을 참고하지 못하도록 룩-어헤드 마스크(look-ahead mask)를 도입했다.
    
2. **Multi-head attention** : h개의 서로 다른 head(각 concept(각 각 Q, K, V로 구성)의 attention 연산 결과)를 만들어 다양한 특징을 학습할 수 있도록 유도 → h개의 각 head의 값과 output matrix와 곱해 계산
    
    → 연산을 수행한 뒤에도 차원은 동일하게 유지된다
    
<img width="623" alt="b" src="https://github.com/k-bum/MachineLearningProject/assets/96854885/90ffece8-59de-45a2-b374-cf31cc3a1838">

I am a student 라는 문장에 대해 I 라는 단어에 대해 self attention 적용 시 Q, K, V는 다음과 같다

- Query : 물어보는 주체(I)
- Key : 물어보는 대상(I, am, a, student)
    
    → I 라는 단어가 전체 문장의 각 단어(Key)에 대해 어떤 가중치 값(attention score, energy)을 가지는지 계산
    
    → mask matrix를 적용해 특정 단어는 무시할 수 있도록 하는 것도 가능하다
    
- Value : 계산된 score(Query와 Keys 간 연관성)와 Value 값을 곱한 후 softmax를 거쳐 실제 attention value를 계산
    
    → encoder-decoder attention의 경우 각 단어를 최종적으로 출력(디코더의 Query)하기 위해 인코더에 어떤 Key와 Value를 참고할지 물어본다
    
<img width="427" alt="c" src="https://github.com/k-bum/MachineLearningProject/assets/96854885/8c9001a8-858a-422c-889c-07afaee7c9ac">

- Scaled dot-product attention : 각 Q벡터는 모든 K벡터에 대해서 어텐션 스코어를 구하고(Q 벡터와 K 벡터를 내적한 후 모델의 입력 차원을 head의 수로 나눈 수의 제곱근 값으로 scaling), softmax 함수를 통해 어텐션 분포를 구한 뒤에 이를 사용하여 모든 V벡터를 가중합하여 어텐션 값을 구하게 된다. 그리고 이를 모든 Q벡터에 대해 반복한다.

→ 이 과정을 행렬 연산으로 한 번에 처리할 수 있고 결과로 어텐션 행렬을 head의 수만큼 얻게 된다. 

![Untitled 1](https://github.com/k-bum/MachineLearningProject/assets/96854885/e0f13949-e704-40c0-bf14-99902c1274dc)

최종적으로 얻은 어텐션 행렬(head)을 연결한 후 최종적으로 가중치 행렬을 곱해 multi-head attention행렬을 얻는다.

<img width="432" alt="d" src="https://github.com/k-bum/MachineLearningProject/assets/96854885/3619cd7a-afd3-4415-8fd8-eb050cf402cc">

![Untitled 2](https://github.com/k-bum/MachineLearningProject/assets/96854885/aed53bc6-645a-45a2-b67f-fb411c97b73b)

![Untitled 3](https://github.com/k-bum/MachineLearningProject/assets/96854885/50d0166f-d49d-47c7-a806-575af92ff060)

성능 향상을 위한 잔여 학습(Residual Learning)을 사용

→ 인코더는 attention과 정규화(normalization) 과정을 반복

디코더의 각 층은 인코더의 마지막 층의 output을 입력으로 받는다

<img width="427" alt="e" src="https://github.com/k-bum/MachineLearningProject/assets/96854885/e409fb11-d7bc-4310-9c3f-4c538d65858c">
