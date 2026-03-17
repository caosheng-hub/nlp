# -*- coding: utf-8 -*-
import torch
from models.dm01_input import Embeddings, PositionEncoding
from models.dm02_encoder import MutiHeadAttention, FeedForward, EncoderLayer, Encoder
from models.dm03_decoder import DecoderLayer, Decoder
from models.dm04_generator import Generator
from models.dm05_transformer import EncoderDecoder

def main():
    print("="*50)
    print("Transformer 论文复现 | 《Attention Is All You Need》")
    print("="*50)

    # 超参数（和原论文完全对齐）
    vocab_size_source = 1000
    vocab_size_target = 2000
    d_model = 512
    head = 8
    d_ff = 2048
    N = 6
    dropout_p = 0.1
    batch_size = 2
    source_seq_len = 4
    target_seq_len = 6

    # 实例化完整模型组件
    base_attn = MutiHeadAttention(embed_dim=d_model, head=head, dropout_p=dropout_p)
    base_ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout_p=dropout_p)

    # 编码器
    encoder_layer = EncoderLayer(size=d_model, self_atten=base_attn, ff=base_ff, dropout_p=dropout_p)
    encoder = Encoder(layer=encoder_layer, N=N)

    # 解码器
    decoder_self_attn = MutiHeadAttention(embed_dim=d_model, head=head, dropout_p=dropout_p)
    decoder_src_attn = MutiHeadAttention(embed_dim=d_model, head=head, dropout_p=dropout_p)
    decoder_ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout_p=dropout_p)
    decoder_layer = DecoderLayer(
        size=d_model,
        self_attn=decoder_self_attn,
        src_attn=decoder_src_attn,
        feed_forward=decoder_ff,
        dropout_p=dropout_p
    )
    decoder = Decoder(layer=decoder_layer, N=N)

    # 嵌入层（词嵌入+位置编码）
    source_embed = torch.nn.Sequential(
        Embeddings(vocab_size=vocab_size_source, d_model=d_model),
        PositionEncoding(d_model=d_model, dropout_p=dropout_p)
    )
    target_embed = torch.nn.Sequential(
        Embeddings(vocab_size=vocab_size_target, d_model=d_model),
        PositionEncoding(d_model=d_model, dropout_p=dropout_p)
    )

    # 输出生成器
    generator = Generator(d_model=d_model, vocab_size=vocab_size_target)

    # 完整Transformer模型
    transformer = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        source_embed=source_embed,
        target_embed=target_embed,
        generator=generator
    )

    print("\n✅ 模型实例化完成，完整结构：")
    print(transformer)

    # 构造测试输入与掩码
    source = torch.tensor([[1, 2, 3, 4], [2, 5, 6, 10]])
    target = torch.tensor([[1, 20, 3, 4, 19, 30], [21, 5, 6, 10, 80, 38]])

    # 测试掩码（全1无遮挡，仅演示用）
    source_mask1 = torch.ones(head, source_seq_len, source_seq_len)
    source_mask2 = torch.ones(head, target_seq_len, source_seq_len)
    target_mask = torch.ones(head, target_seq_len, target_seq_len)

    # 模型前向传播
    print("\n🚀 执行模型前向传播...")
    with torch.no_grad():
        output = transformer(source, target, source_mask1, source_mask2, target_mask)

    # 输出结果
    print("\n✅ 运行成功！")
    print(f"📌 模型输出形状: {output.shape}")
    print(f"   对应维度: [batch_size={batch_size}, target_seq_len={target_seq_len}, target_vocab_size={vocab_size_target}]")
    print(f"📌 输出概率示例（首句前2个token的前5个词表概率）:\n{output[0, :2, :5]}")
    print("\n" + "="*50)
    print("🎉 Transformer 论文复现代码验证完成！")
    print("="*50)

if __name__ == "__main__":
    main()
