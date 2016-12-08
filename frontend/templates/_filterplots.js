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

function collect_plot_options() {
  var opts = {
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
    alpha: $("#plt_alpha").val(),
    clear: +$("#plt_clear").is(":checked"),
    legend: +$("#plt_legend").is(":checked"),
    line_width: $("#plt_lw").val(),
    cmap: $("#plt_cmap").val(),
    chan_mask: +$("#chan_mask").is(":checked"),
    pp: collect_pp_args($('#pp_options')),
  };
  return add_baseline_args($('#blr_options'), opts);
}

function do_filtered_plot() {
  $('#plot_button>.dots').text("ting...").fadeIn();
  var post_data = collect_plot_options();
  var ds_info = collect_ds_info();
  post_data['ds_kind'] = ds_info.kind;
  post_data['ds_name'] = ds_info.name;
  post_data['fignum'] = fig.id;

  var cbs = make_post_callbacks('#plot_button>.dots');
  $.ajax({
    url: '/_filterplot',
    type: 'POST',
    data: post_data,
    dataType: 'json',
    success: cbs['success'],
    error: cbs['fail'],
  });
}

function download_spectrum_data() {
  var ds_info = collect_ds_info();
  var args = {
    ds_name: ds_info.name,
    ds_kind: ds_info.kind,
    meta_keys: $('#dl_metadata option:selected').map(_value).toArray(),
  };
  window.open('/'+fig.id+'/spectra.csv?' + $.param(args), '_blank');
}
