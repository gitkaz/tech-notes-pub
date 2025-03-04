# mlx_lmにおけるKVキャッシュ量子化の実装に関するmemo
**2024/2/10 時点の情報です**

**概要:**

[mlx_lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)におけるKVキャッシュ量子化の実装について


## 1. `generate_step` 関数とKVキャッシュ関連の引数

[mlx_lm.utils.generate_step](https://github.com/ml-explore/mlx-examples/blob/1ced1b00ca9c2457fcbf0e54ffcffe58f53fb4fd/llms/mlx_lm/utils.py#L209) 関数は、MLXフレームワークでテキスト生成を行う際の主要な関数であり、以下のKVキャッシュに関連する引数を受け取る。これらの引数は、KVキャッシュの量子化を制御するために使用される。

*   **`prompt_cache`**: 量子化の対象となるKVキャッシュ
    *   簡単に型を表すなら `prompt_cache: list[KVCache | QuantizedKVCache]`
    *   そもそも、KVキャッシュは、LLMの各レイヤーにおけるAttentionメカニズムで計算されたKey (K) と Value (V) のテンソルを格納しすることで、計算を効率化するもの。
    *   prompt_cacheにはモデルの各レイヤーごとのKV Cacheがリストで保存される or されている（先頭のレイヤーのキャッシュはpropt_cache[0]）。
    *   リスト内の各要素は `KVCache` または `QuantizedKVCache` オブジェクト
*   **`kv_bits`**: KVキャッシュを量子化する際のビット数
*   **`kv_group_size`**: KVキャッシュの量子化におけるグループサイズを指定する
    *   kv_group_size の単位で量子化が行われる。小さいと精度が高くなり、一方でキャッシュ生成時のメモリ使用量と計算量が増加する。大きい場合はその逆。デフォルトは64
*   **`quantized_kv_start`**: KVキャッシュを量子化するかを指定するステップ数
    *   ステップ数(トークン数)が `quantized_kv_start` 以下の場合、KVキャッシュは量子化されない。`maybe_quantize_kv_cache` の処理を参照。

## 2. `maybe_quantize_kv_cache` 関数について

[maybe_quantize_kv_cache](https://github.com/ml-explore/mlx-examples/blob/1ced1b00ca9c2457fcbf0e54ffcffe58f53fb4fd/llms/mlx_lm/utils.py#L196) 関数は、`generate_step` 関数から呼び出され、与えられた条件に基づいてKVキャッシュ (prompt_cache) を量子化する役割を担っている。

```python
def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if (
        kv_bits is not None
        and not isinstance(prompt_cache[0], cache.QuantizedKVCache)
        and prompt_cache[0].offset > quantized_kv_start
    ):
        for i in range(len(prompt_cache)):
            if isinstance(prompt_cache[i], cache.KVCache):
                prompt_cache[i] = prompt_cache[i].to_quantized(
                    group_size=kv_group_size, bits=kv_bits
                )
```

*   **処理の流れ:**
    1.  量子化の条件確認:
        *   `kv_bits is not None`: ユーザによる引数で、量子化が有効になっているか。
        *   `not isinstance(prompt_cache[0], cache.QuantizedKVCache)`: KVキャッシュが既に量子化されていないか。
        *   `prompt_cache[0].offset > quantized_kv_start`: 現在のステップ数（トークン数）が、量子化を開始する閾値を超えているか。
    2.  KVキャッシュの量子化: 上記の条件を満たす場合に、`prompt_cache` の各要素 (KVCache オブジェクト) に対して、`to_quantized` メソッドを呼び出して量子化を行う。

## 3. 関連クラス

KV Cacheは、量子化と未量子化でクラスが分かれている。以下のクラスは、[mlx_lm.models.cache.py](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/cache.py) に定義されている。

*   **`KVCache`:** 量子化されていない（元の精度、例えば FP16 の）KVキャッシュを格納するクラス。
*   **`QuantizedKVCache`:** 量子化されたKVキャッシュを格納するクラス。

*   どちらにも `offset` 属性があり、ここに保持している有効なトークン数が記録されている。
*   KVCacheクラスには [to_quantized](https://github.com/ml-explore/mlx-examples/blob/1ced1b00ca9c2457fcbf0e54ffcffe58f53fb4fd/llms/mlx_lm/models/cache.py#L268)というメソッドがある。
    *   KVCacheオブジェクトを QuantizedKVCache オブジェクトに変換するメソッド。引数でグループサイズとビット数を指定して量子化する。

```python
    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None:
            quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self.values, group_size=group_size, bits=bits
            )
        return quant_cache
```
