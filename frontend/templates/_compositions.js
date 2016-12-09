function plot_compositions(btn) {
  var err_span = $(btn).next('.err_msg');

  var xc = $('#x_comp_options option:selected').map(_value).toArray();
  var yc = $('#y_comp_options option:selected').map(_value).toArray();
  if (xc.length + yc.length == 0) {
    err_span.text('Please select at least one composition to plot.').show();
    return;
  }
  var ds_info = collect_ds_info();
  if (ds_info.name.length > 1) {
    err_span.text('Too many datasets selected.').show();
    return;
  }
  var post_data = {
    x_comps: xc.join('+'),
    y_comps: yc.join('+'),
    color_by: $('#color_by > option:selected').val(),
    do_fit: +$('#do_fit').is(':checked'),
    use_mols: +$('#use_mols').is(':checked'),
    fignum: fig.id,
    ds_kind: ds_info.kind[0],
    ds_name: ds_info.name[0],
  };
  add_plot_args(post_data);

  var wait = $('.wait', btn).show();
  $.ajax({
    url: '/_plot_compositions',
    type: 'POST',
    data: post_data,
    dataType: 'json',
    success: function(data) {
      wait.hide();
      err_span.hide();
      $('#fit_info td:last-child span').empty();
      update_zoom_ctrl(data['zoom']);
      delete data['zoom'];
      for (key in data) {
        $('#fit_' + key).text(data[key].toFixed(3));
      }
    },
    error: function(jqXHR, textStatus, errorThrown) {
      wait.hide();
      err_span.text(jqXHR.responseText).show();
    }
  });
}

function download_composition_data() {
  var ds_info = collect_ds_info();
  var args = { ds_name: ds_info.name[0], ds_kind: ds_info.kind[0] };
  window.open('/'+fig.id+'/compositions.csv?' + $.param(args), '_blank');
}
