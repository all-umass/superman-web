function run_preds(btn) {
  _run_model(btn, collect_ds_info(), false);
}
function run_crossval(btn) {
  _run_model(btn, collect_ds_info(), null);
}
function train_model(btn) {
  _run_model(btn, collect_ds_info(), true);
}
function upload_model(btn) {
  var ds_info = collect_ds_info(),
      ds_kind = ds_info.kind[0],
      file_input = $('#modelfile')[0],
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
      $('.needs_model').attr('disabled', false);
      $('#model_info').html(JSON.parse(data).info);
      $('#model_error>tbody').empty();
    }
  });
}
function _run_model(btn, ds_info, do_train) {
  var target_meta = $('#ds_prediction_container .target_meta');
  var pred_vars = $('option:selected', target_meta).map(_value).toArray();
  var post_data = {
      ds_name: ds_info.name,
      ds_kind: ds_info.kind,
      fignum: fig.id,
      pp: collect_pp_args($('#pp_options')),
      pls_comps: +$('#pls_comps').val(),
      pls_kind: $('#pls_kind').val(),
      pred_meta: pred_vars,
      cv_folds: +$('#cv_folds').val(),
      cv_min_comps: +$('#cv_min_comps').val(),
      cv_max_comps: +$('#cv_max_comps').val(),
  };
  add_resample_args($('#resample_options'), post_data);
  add_baseline_args($('#blr_options'), post_data);
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
    url: '/_run_model',
    data: post_data,
    dataType: 'json',
    success: function(data){
      wait.hide();
      if (!do_train) return;
      $('.needs_model').attr('disabled', false);
      var stats = data.stats;
      var tbody = $('#model_error>tbody').empty();
      for (var i=0; i<stats.length; i++) {
        var v = stats[i];
        tbody.append('<tr><td>'+v.name+'</td><td>' + v.r2.toPrecision(3) +
                     '</td><td>' + v.rmse.toPrecision(4) + '</td></tr>');
      }
      $('#model_info').html(data.info);
    },
    error: function(jqXHR, textStatus, errorThrown) {
      wait.hide();
      err_span.text(jqXHR.responseText);
    }
  });
}
function plot_coefs(btn) {
  var ds_info = collect_ds_info();
  var post_data = {
      ds_name: ds_info.name,
      ds_kind: ds_info.kind,
      fignum: fig.id,
      pp: collect_pp_args($('#pp_options')),
  };
  add_resample_args($('#resample_options'), post_data);
  add_baseline_args($('#blr_options'), post_data);
  add_plot_args(post_data);
  var wait = $('.wait', btn).show();
  var err_span = $(btn).next('.err_msg').empty();
  $.ajax({
    type: 'POST',
    url: '/_plot_model_coefs',
    data: post_data,
    success: function(){
      wait.hide();
    },
    error: function(jqXHR, textStatus, errorThrown) {
      wait.hide();
      err_span.text(jqXHR.responseText);
    }
  });
}
function predict_download(btn) {
  var dl_type = $(btn).next('select').val();
  var dl_url = '/'+fig.id+'/pls_';
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
}
