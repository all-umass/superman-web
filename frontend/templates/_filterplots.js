function _value() {
  return this.value;
}

function changed(kind, val) {
  $('td.'+val).hide();
  $('td.'+val+'.'+kind).show();
}

function child_vals(selector){
  return "[" + $(selector).children().map(function(i,x){
    return x.value;
  }).filter(function(i,x){
    return x;
  }).toArray() + "]";
}

function do_filtered_plot(btn) {
  var ds_info = collect_ds_info();
  var post_data = {
    xaxis: $("#xaxis").val(),
    yaxis: $("#yaxis").val(),
    color: $("#color").val(),
    x_metadata: $("td.x.metadata").children().val(),
    x_line_ratio: child_vals("td.x.line_ratio"),
    x_computed: $("td.x.computed").children().val(),
    y_metadata: $("td.y.metadata").children().val(),
    y_line_ratio: child_vals("td.y.line_ratio"),
    y_computed: $("td.y.computed").children().val(),
    fixed_color: $("td.color.default").children().val(),
    color_by: $("td.color.metadata").children().val(),
    color_line_ratio: child_vals("td.color.line_ratio"),
    color_computed: $("td.color.computed").children().val(),
    chan_mask: +$("#chan_mask").is(":checked"),
    pp: collect_pp_args($('#pp_options')),
    ds_kind: ds_info.kind,
    ds_name: ds_info.name,
    fignum: fig.id,
  };
  add_plot_args(post_data);
  add_resample_args($('#resample_options'), post_data);
  add_baseline_args($('#blr_options'), post_data);

  var err_span = $(btn).next('.err_msg');
  var wait = $('.wait', btn).show();
  $.ajax({
    url: '/_filterplot',
    type: 'POST',
    data: post_data,
    dataType: 'json',
    success: function(data) {
      wait.hide();
      err_span.hide();
      update_zoom_ctrl(data);
    },
    error: function(jqXHR, textStatus, errorThrown) {
      wait.hide();
      err_span.text(jqXHR.responseText).show();
    }
  });
}

function download_spectrum_data() {
  var ds_info = collect_ds_info();
  var args = {
    ds_name: ds_info.name,
    ds_kind: ds_info.kind,
    meta_keys: $('#dl_metadata option:selected').map(_value).toArray(),
    as_matrix: +$('#dl_vector').is(':checked'),
  };
  window.open('/'+fig.id+'/spectra.csv?' + $.param(args), '_blank');
  if (args['meta_keys'].length > 0 || args['ds_name'].length > 1) {
    window.open('/'+fig.id+'/metadata.csv?' + $.param(args), '_blank')
  }
}
