var Classify = (function() {
  function _run_model(btn, ds_info, do_train) {
    var container = $('#ds_classification_container');
    var err_span = $(btn).parents('table,div').first().find('.err_msg').empty();
    var pred_var = $('.target_meta option:selected', container).val();
    if (pred_var == null && do_train !== false) {
      err_span.text('No variables selected.');
      return;
    }
    var post_data = {
        ds_name: ds_info.name,
        ds_kind: ds_info.kind,
        fignum: fig.id,
        pp: collect_pp_args($('#pp_options')),
        pred_var: pred_var,
        model_kind: $('.model_kind', container).val(),
        knn_k: +$('.knn_k', container).val(),
        logistic_C: +$('.logistic_C', container).val(),
        cv_folds: +$('.cv_folds', container).val(),
        cv_stratify: $('.cv_stratify', container).val(),
        cv_min_k: +$('.cv_min_k', container).val(),
        cv_max_k: +$('.cv_max_k', container).val(),
        cv_min_logC: +$('.cv_min_C', container).val(),
        cv_max_logC: +$('.cv_max_C', container).val(),
    };
    add_resample_args($('#resample_options'), post_data);
    add_baseline_args($('#blr_options'), post_data);
    if (do_train !== null) {
      post_data['do_train'] = +do_train;
    }
    var wait = $('.wait', btn).show();
    $.ajax({
      type: 'POST',
      url: '/_run_classifier',
      data: post_data,
      dataType: 'json',
      success: function(data){
        wait.hide();
        if (do_train === null) return;
        $('.needs_model', container).attr('disabled', false);
        $('.model_info', container).html(data.info);
      },
      error: function(jqXHR, textStatus, errorThrown) {
        wait.hide();
        err_span.text(jqXHR.responseText);
      }
    });
  }

  return {
    predict: function(btn) {
      _run_model(btn, collect_ds_info(), false);
    },
    crossval: function(btn) {
      _run_model(btn, collect_ds_info(), null);
    },
    train: function(btn) {
      _run_model(btn, collect_ds_info(), true);
    },
    upload: function(btn) {
      var container = $('#ds_classification_container');
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
        setTimeout(function() { td.css('animation', ''); }, 1000);
        return;
      }
      var post_data = new FormData();
      post_data.append('fignum', fig.id);
      post_data.append('modelfile', f);
      post_data.append('ds_kind', ds_kind);
      post_data.append('model_type', 'classifier');
      var wait = $('.wait', btn).show();
      $.ajax({
        url: '/_load_model',
        data: post_data,
        processData: false,
        contentType: false,
        type: 'POST',
        error: function(jqXHR, textStatus, errorThrown) {
          wait.hide();
          file_input.value = '';  // reset the input
          err_span.text(jqXHR.responseText);
        },
        success: function(data) {
          wait.hide();
          file_input.value = '';  // reset the input
          $('.needs_model', container).attr('disabled', false);
          $('.model_info', container).html(JSON.parse(data).info);
        }
      });
    },
    download: function(btn) {
      var dl_type = $(btn).next('select').val();
      var dl_url = '/'+fig.id+'/classifier_';
      if (dl_type === 'preds') {
        var ds_info = collect_ds_info();
        dl_url += 'predictions.csv?' + $.param({ds_name: ds_info.name,
                                                ds_kind: ds_info.kind});
      } else if (dl_type === 'pkl') {
        dl_url += 'model.pkl';
      } else {
        dl_url += 'model.csv';
      }
      window.open(dl_url, '_blank');
    },
    change_kind: function(option) {
      var container = $('#ds_classification_container');
      switch (option.value) {
        case 'logistic':
          $('.for_knn', container).hide();
          $('.for_logistic', container).show();
          break;
        case 'knn':
          $('.for_logistic', container).hide();
          $('.for_knn', container).show();
          break;
      }
    },
  };
})();
