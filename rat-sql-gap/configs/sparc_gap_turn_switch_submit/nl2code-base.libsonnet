# Model details:
# - NL2Code
# - Pretrained, fixed word embeddings
#   - glove-42B
#   - min_freq 50
# - Spiderv2 encoder
#   - question_encoder ['emb', 'bilstm']
#   - column_encoder ['emb', 'bilstm-summarize']
#   - table_encoder ['emb', 'bilstm-summarize']
#   - upd_steps 4
# - Optimization
#   - max_steps 40k
#   - batch_size 10
#   - Adam with lr 1e-3

function(output_from, data_path='data/sparc-20190205/') {
    local PREFIX = data_path,

    data: {
        trn: {
            name: 'spider',
            paths: ['sparc_preproc/train.json'],
            tables_paths: ['raw_data/sparc/tables.json'],
            db_path: 'raw_data/sparc/database',
        },
        val: {
            name: 'spider',
            paths: ['sparc_preproc/dev.json'],
            tables_paths: ['raw_data/sparc/tables.json'],
            db_path: 'raw_data/sparc/database',
        },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'spiderv2',
            dropout: 0.2,
            word_emb_size: 300,
            question_encoder: ['emb', 'bilstm'],
            column_encoder: ['emb', 'bilstm-summarize'],
            table_encoder: ['emb', 'bilstm-summarize'],
            update_config:  {
                name: 'relational_transformer',
                num_layers: 4,
                num_heads: 8,
            },
        },
        decoder: {
            name: 'NL2Code',
            dropout: 0.2,
            desc_attn: 'mha',
        },
        encoder_preproc: {
            word_emb: {
                name: 'glove',
                kind: '42B',
            },
            count_tokens_in_word_emb_for_vocab: false,
            min_freq: 50,
            max_count: 5000,
            include_table_name_in_column: false,

            save_path: PREFIX + 'sparc,nl2code-0401,output_from=%s,emb=glove-42B,min_freq=50/' % [output_from],
        },
        decoder_preproc: self.encoder_preproc {
            grammar: {
                name: 'spider',
                output_from: output_from,
                use_table_pointer: output_from,
                include_literals: false,
            },
            use_seq_elem_rules: true,

            word_emb:: null,
            include_table_name_in_column:: null,
            count_tokens_in_word_emb_for_vocab:: null,
        },
    },

    train: {
        batch_size: 10,
        eval_batch_size: 50,

        keep_every_n: 500,
        eval_every_n: 1000000,
        save_every_n: 500,
        report_every_n: 100,

        max_steps: 41000,
        num_eval_items: 50,
    },
    optimizer: {
        name: 'adam',
        lr: 0.0,
    },
    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: $.train.max_steps / 20,
        start_lr: 1e-3,
        end_lr: 0,
        decay_steps: $.train.max_steps - self.num_warmup_steps,
        power: 0.5,
    }
}
