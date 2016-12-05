function run_preds(btn) {
  _run_model(btn, collect_ds_info(), false);
}
function run_crossval(btn) {
  _run_model(btn, collect_ds_info(), null);
}
function make_model(btn) {
  var ds_info = collect_ds_info();
  var file_input = $('#modelfile')[0];
  if (file_input.files.length > 0) {
    _upload_model(btn, file_input, ds_info.kind[0]);
  } else {
    _run_model(btn, ds_info, true);
  }
}
function _upload_model(btn, file_input, ds_kind) {
  var f = file_input.files[0];
  var err_span = $(btn).next('.err_msg');
  if (f.size > 5000000) {
    err_span.text('File too big (max 5 MB)').show();
    // flash the file input's enclosing <td>
    var td = $(file_input).parent().css('animation', 'flash 1s');
    setTimeout(function() { td.css('animation', ''); }, 1000);
    return
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
      err_span.text(jqXHR.responseText).show();
    },
    success: function(data) {
      wait.hide();
      err_span.hide();
      file_input.value = '';  // reset the input
      $('.needs_model').attr('disabled', false);
      $('#model_info').html(JSON.parse(data).info);
      $('#model_error>tbody').empty();
    }
  });
}
function _run_model(btn, ds_info, do_train) {
  var pred_vars = $('#target_meta option:selected').map(_value).toArray();
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
  add_baseline_args($('#blr_options'), post_data);
  if (do_train !== null) {
    post_data['do_train'] = +do_train;
  }

  var msg = $('#target_meta').nextAll('.err_msg');
  if (pred_vars.length == 0 && do_train) {
    msg.show('highlight');
    var box = $('#target_meta').closest('.box');
    box.css('animation', 'flash 1s');
    setTimeout(function() { box.css('animation', ''); }, 1000);
    return;
  } else {
    msg.hide();
  }
  var wait = $('.wait', btn).show();
  var err_span = $(btn).next('.err_msg');
  $.ajax({
    type: 'POST',
    url: '/_run_model',
    data: post_data,
    dataType: 'json',
    success: function(data){
      wait.hide();
      err_span.hide();
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
      err_span.text(jqXHR.responseText).show();
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
