var Regress = (function () {
    function _run_model(btn, ds_info, do_train) {
        var container = $('#ds_prediction_container');
        var pred_vars = multi_val($('.target_meta option:selected', container));
        var post_data = {
            ds_name: ds_info.name,
            ds_kind: ds_info.kind,
            fignum: fig.id,
            pp: GetArgs.pp($('#pp_options')),
            pls_comps: +$('.pls_comps', container).val(),
            lasso_alpha: +$('.lasso_alpha', container).val(),
            lars_num_channels: +$('.lars_num_channels', container).val(),
            regress_kind: $('.model_kind', container).val(),
            variate_kind: $('.variate_kind', container).val(),
            pred_meta: pred_vars,
            cv_folds: +$('.cv_folds', container).val(),
            cv_stratify: $('.cv_stratify', container).val(),
            cv_min_comps: +$('.cv_min_comps', container).val(),
            cv_max_comps: +$('.cv_max_comps', container).val(),
            cv_min_chans: +$('.cv_min_chans', container).val(),
            cv_max_chans: +$('.cv_max_chans', container).val(),
        };
        GetArgs.resample($('#resample_options'), post_data);
        GetArgs.baseline($('#blr_options'), post_data);
        if (do_train !== null) {
            post_data['do_train'] = +do_train;
        }

        var err_span = $(btn).parents('table,div').first().find('.err_msg').empty();
        if (pred_vars.length == 0 && do_train !== false) {
            err_span.text('No variables selected.');
            return;
        }
        var wait = $('.wait', btn).show();
        $.ajax({
            type: 'POST',
            url: '/_run_regression',
            data: post_data,
            dataType: 'json',
            success: function (data) {
                wait.hide();
                if (do_train === null) return;
                $('.needs_model', container).attr('disabled', false);
                var stats = data.stats;
                var tbody = $('.model_error>tbody', container).empty();
                var show_table = false;
                for (var i = 0; i < stats.length; i++) {
                    var v = stats[i];
                    if (v.r2 !== null && v.rmse !== null) {
                        tbody.append('<tr><td>' + v.name + '</td><td>' + v.r2.toPrecision(3) +
                            '</td><td>' + v.rmse.toPrecision(4) + '</td></tr>');
                        show_table = true;
                    }
                }
                $('.model_info', container).html(data.info);
                $('.model_error', container).toggle(show_table);
            },
            error: function (jqXHR, textStatus, errorThrown) {
                wait.hide();
                err_span.text(jqXHR.responseText);
            }
        });
    }

    return {
        predict: function (btn) {
            _run_model(btn, collect_ds_info(), false);
        },
        crossval: function (btn) {
            _run_model(btn, collect_ds_info(), null);
        },
        train: function (btn) {
            _run_model(btn, collect_ds_info(), true);
        },
        upload: function (btn) {
            var container = $('#ds_prediction_container');
            var ds_info = collect_ds_info(),
                ds_kind = ds_info.kind[0],
                file_input = $('.modelfile', container)[0],
                err_span = $(btn).closest('table').find('.err_msg').empty(),
                do_flash = false;
            if (file_input.files.length != 1) {
                err_span.text('No file selected');
                do_flash = true;
            } else {
                var f = file_input.files[0];
                if (f.size > 5000000) {
                    err_span.text('File too big (max 5 MB)');
                    do_flash = true;
                }
            }
            if (do_flash) {
                // flash the file input's enclosing <td> and return
                var td = $(file_input).parent().css('animation', 'flash 1s');
                setTimeout(function () {
                    td.css('animation', '');
                }, 1000);
                return;
            }
            var post_data = new FormData();
            post_data.append('fignum', fig.id);
            post_data.append('modelfile', f);
            post_data.append('ds_kind', ds_kind);
            post_data.append('model_type', 'regression');
            var wait = $('.wait', btn).show();
            $.ajax({
                url: '/_load_model',
                data: post_data,
                processData: false,
                contentType: false,
                type: 'POST',
                error: function (jqXHR, textStatus, errorThrown) {
                    wait.hide();
                    file_input.value = '';  // reset the input
                    err_span.text(jqXHR.responseText);
                },
                success: function (data) {
                    wait.hide();
                    file_input.value = '';  // reset the input
                    $('.needs_model', container).attr('disabled', false);
                    $('.model_info', container).html(JSON.parse(data).info);
                    $('.model_error>tbody', container).empty();
                }
            });
        },
        plot_coefs: function (btn) {
            var ds_info = collect_ds_info();
            var post_data = {
                ds_name: ds_info.name,
                ds_kind: ds_info.kind,
                fignum: fig.id,
                pp: GetArgs.pp($('#pp_options')),
            };
            GetArgs.resample($('#resample_options'), post_data);
            GetArgs.baseline($('#blr_options'), post_data);
            GetArgs.plot(post_data);
            var wait = $('.wait', btn).show();
            var err_span = $(btn).next('.err_msg').empty();
            $.ajax({
                type: 'POST',
                url: '/_plot_model_coefs',
                data: post_data,
                success: function () {
                    wait.hide();
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    wait.hide();
                    err_span.text(jqXHR.responseText);
                }
            });
        },
        download: function (btn) {
            var dl_type = $(btn).next('select').val();
            var dl_url = '/' + fig.id + '/regression_';
            if (dl_type === 'preds') {
                var ds_info = collect_ds_info();
                dl_url += 'predictions.csv?' + $.param({
                    ds_name: ds_info.name,
                    ds_kind: ds_info.kind
                });
            } else {
                dl_url += 'model.' + dl_type;
            }
            window.open(dl_url, '_blank');
        },
        change_kind: function (option) {
            var container = $('#ds_prediction_container');
            switch (option.value) {
                case 'lasso':
                    $('.for_pls,.for_lars', container).hide();
                    $('.for_lasso', container).show();
                    break;
                case 'pls':
                    $('.for_lasso,.for_lars', container).hide();
                    $('.for_pls', container).show();
                    break;
                case 'lars':
                    $('.for_lasso,.for_pls', container).hide();
                    $('.for_lars', container).show();
                    break;
            }
        },
    };
})();
