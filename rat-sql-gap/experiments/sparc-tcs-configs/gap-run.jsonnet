{
    local exp_id = 5,
    logdir: "checkpoint/",
    model_config: "configs/sparc_gap_turn_switch_submit/gap-bart.jsonnet",
    model_config_args: {
        bs: 12,
        num_batch_accumulated: 2,
        bart_version: "facebook/bart-large",
        summarize_header: "avg",
        use_column_type: false,
        num_layers: 8,
        lr: 1e-4,
        bert_lr: 1e-5,
        att: 1,
        end_lr: 0,
        sc_link: true,
        cv_link: true,
        use_align_mat: true,
        use_align_loss: true,
        use_tc_loss: false,
        bart_token_type: true,
        decoder_hidden_size: 512,
        end_with_from: true, # equivalent to "SWGOIF" if true
        clause_order: null, # strings like "SWGOIF", it will be prioriotized over end_with_from
    },

    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [37500],
    eval_section: "val",
}
