var Matching = (function () {
    return {
        search: function (btn) {
            var err_span = $(btn).next('.err_msg');
            if ($('.needs_plot', btn).attr('disabled')) {
                err_span.text('Choose a query spectrum.');
                return;
            }
            // make sure we run the final pp before matching
            SingleSpectrum.preprocess($('#query_prep>table'));
            // collect params
            var ds_info = collect_ds_info();
            var post_data = {
                ds_name: ds_info.name,
                ds_kind: ds_info.kind,
                pp: GetArgs.pp($('#pp_options')),
                metric: $('#wsm_metric > option:selected').val(),
                param: $('#wsm_param').text(),
                min_window: $('#wsm_min_window').text(),
                num_results: $('#wsm_num_results').val(),
                num_comps: $('#wsm_endmembers').val(),
                score_pct: $('#wsm_score_pct').text(),
                fignum: fig.id
            };
            GetArgs.resample($('#resample_options'), post_data);
            GetArgs.baseline($('#blr_options'), post_data);
            // match
            var res = $('#wsm_results').fadeOut(),
                wait = $('.wait', btn).show();
            $.ajax({
                url: '/_spectrum_matching',
                type: 'POST',
                data: post_data,
                success: function (data) {
                    wait.hide();
                    err_span.empty();
                    res.html(data).fadeIn();
                    $('button', res).prop('disabled', false);
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    wait.hide();
                    err_span.text(jqXHR.responseText);
                    $('button', res).prop('disabled', true);
                }
            });
        },
        compare: function (names, target_name, target_kind) {
            var post_data = {
                compare: JSON.stringify(names),
                target_name: target_name,
                target_kind: target_kind,
                pp: GetArgs.pp($('#pp_options')),
                fignum: fig.id
            };
            GetArgs.resample($('#resample_options'), post_data);
            GetArgs.baseline($('#blr_options'), post_data);
            $.post('/_compare', post_data);
        },
        download: function () {
            window.open('/' + fig.id + '/search_results.csv', '_blank');
        },
    };
})();
