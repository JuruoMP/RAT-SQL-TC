local _0428_base = import 'nl2code-base.libsonnet';
local _data_path = 'data/sparc-bart-tcs/';
local _output_from = true;
local _fs = 2;

function(args) _0428_base(output_from=_output_from, data_path=_data_path) + {
    local lr_s = '%0.1e' % args.lr,
    local bert_lr_s = '%0.1e' % args.bert_lr,
    local end_lr_s = if args.end_lr == 0 then '0e0' else '%0.1e' % args.end_lr,

    local base_bert_enc_size = 1024,
    local enc_size =  base_bert_enc_size,

    model+: {
        encoder+: {
            name: 'spider-history-bart',
            batch_encs_update:: null,
            question_encoder:: null,
            column_encoder:: null,
            table_encoder:: null,
            dropout:: null,
            update_config+:  {
                name: 'relational_transformer',
                num_layers: args.num_layers,
                num_heads: 8,
                sc_link: args.sc_link,
                cv_link: args.cv_link,
                max_segment_id: 0,
            },
            summarize_header: args.summarize_header,
            use_column_type: args.use_column_type,
            bart_version: args.bart_version,
            top_k_learnable:: null,
            word_emb_size:: null,
            turn_switch_config+:  {
                    model:'turn-switch-classifier-interact',
                    vocab_path: "static/turn_switch_label_vocab.json",
                    reg_loss_scalar:0,
                    hidden_dim: 1024,
                    dropout: 0.10,
                    leaky_rate:0.20,
                    loss_scalar:4.0,
                    mid_layer_activator:'relu',
                    use_dynamic_loss_weight:0
            },
            turn_switch_col_config+:  {
                    model:'turn-switch-col-classifier',
                    vocab_path: "static/turn_switch_col_vocab.json",
                    hidden_dim: 1024,
                    dropout: 0.10,
                    loss_scalar:4.0,
                    use_dynamic_loss_weight:0,
                    use_pre_turn_diff:0
            },
        },
        encoder_preproc+: {
            word_emb:: null,
            min_freq:: null,
            max_count:: null,
            db_path: "raw_data/sparc/database",
            compute_sc_link: args.sc_link,
            compute_cv_link: args.cv_link,
            fix_issue_16_primary_keys: true,
            bart_version: args.bart_version,
            count_tokens_in_word_emb_for_vocab:: null,
            save_path: _data_path + 'sparc,nl2code-1115,output_from=%s,fs=%d,emb=bart,cvlink' % [_output_from, _fs],
        },
        decoder_preproc+: {
            grammar+: {
                end_with_from: args.end_with_from,
                clause_order: args.clause_order,
                infer_from_conditions: true,
                factorize_sketch: _fs,
            },
            save_path: _data_path + 'sparc,nl2code-1115,output_from=%s,fs=%d,emb=bart,cvlink' % [_output_from, _fs],

            compute_sc_link:: null,
            compute_cv_link:: null,
            db_path:: null,
            fix_issue_16_primary_keys:: null,
            bart_version:: null,
        },
        decoder+: {
            name: 'NL2Code-history',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            enc_recurrent_size: enc_size,
            recurrent_size : args.decoder_hidden_size,
            loss_type: 'label_smooth',
            use_align_mat: args.use_align_mat,
            use_align_loss: args.use_align_loss,
            use_tc_loss:false
        },
    },

    train+: {
        batch_size: args.bs,
        num_batch_accumulated: args.num_batch_accumulated,
        clip_grad: 1,

        model_seed: args.att,
        data_seed:  args.att,
        init_seed:  args.att,
    },

    optimizer: {
        name: 'bertAdamw',
        lr: 0.0,
        bert_lr: 0.0,
    },

    lr_scheduler+: {
        name: 'bert_warmup_polynomial_group',
        start_lrs: [args.lr, args.bert_lr],
        end_lr: args.end_lr,
        num_warmup_steps: $.train.max_steps / 8,
    },

}
