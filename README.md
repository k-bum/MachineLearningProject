# MachineLearningProject
Stock price forecast using Transformer

##Transformer
Transformer : Attention is all you need
<기존 Seq2Seq 모델의 한계점>
- 고정된 크기의 context vector에 문장의 정보를 압축하는 과정에서 병목(bottle neck)이 발생해 성능 하락의 원인이 된다
- 새로운 단어가 입력될 때마다 hidden state 값을 갱신 → 이전의 정보를 모두 포함하고 있다 → 마지막 hidden state 값을 모든 단어의 정보를 포함하고 있는 context vector로 설정
- 문장의 길이가 긴 경우 처리하는 것이 어렵다(앞에 위치한 단어의 정보가 encoding 과정에서 손실된다)
- 디코더가 context vector를 매번 참고하면 성능을 기존보다 향상할 수는 있으나 context vector가 모든 문장의 정보를 가지고 있어야 하므로 성능 저하의 원인이 된다
→ 매번 인코더의 모든 출력(hidden state)을 디코더에서 참고한다(attention mechanism)
디코더는 매번 인코더의 모든 출력 중에서 어떤 정보가 중요한지를 계산(energy) → energy 값에 softmax를 적용해 가중치 값을 인코더의 hidden state 값에 적용해 사용

energy는 인코더의 모든 출력값들 중에서 어떤 값과 연관성이 가장 높은지 계산한 값이다 → attention 가중치를 사용해 각 출력이 어떤 입력 정보를 참고했는지 알 수 있다

transformer는 RNN, CNN을 전혀 사용하지 않는다(인코더와 디코더를 여러 개 사용) → 문장 내 각 단어들의 순서에 대한 정보를 주기 위해서 Positional encoding(위치 정보를 포함하고 있는 embedding)을 사용

Positional encoding : transformer는 단어 입력을 순차적으로 받는 방식이 아니기 때문에 단어의 위치 정보를 알려줘야 하므로 각 단어의 embedding vector에 위치 정보를 더해 모델의 입력으로 사용한다 → 같은 단어라고 하더라도 문장 내의 위치에 따라 입력으로 들어가는 embedding vector의 값이 달라진다
