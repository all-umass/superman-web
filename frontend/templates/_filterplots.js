var Plot = (function() {
  function child_vals(selector){
    return "[" + $(selector).children().map(function(i,x){
      return x.value;
    }).filter(function(i,x){
      return x;
    }).toArray() + "]";
  }

  return {
    plot: function(btn) {
      var is_fig_indp = btn.id=="ind_plot_button" ? true : false;
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
        pp: GetArgs.pp($('#pp_options')),
        ds_kind: ds_info.kind,
        ds_name: ds_info.name,
        fignum: fig.id,
        figindp: is_fig_indp
      };
      GetArgs.plot(post_data);
      GetArgs.resample($('#resample_options'), post_data);
      GetArgs.baseline($('#blr_options'), post_data);

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
          alert(errorThrown)
          err_span.text(jqXHR.responseText).show();
        }
      });
    },
    download: function() {
      var ds_info = collect_ds_info();
      var args = {
        ds_name: ds_info.name,
        ds_kind: ds_info.kind,
        meta_keys: multi_val($('#dl_metadata option:selected')),
        as_matrix: +$('#dl_vector').is(':checked'),
      };
      window.open('/'+fig.id+'/spectra.csv?' + $.param(args), '_blank');
      if (args['meta_keys'].length > 0 || args['ds_name'].length > 1) {
        window.open('/'+fig.id+'/metadata.csv?' + $.param(args), '_blank')
      }
    },
    changed: function(kind, val) {
      $('td.'+val).hide();
      $('td.'+val+'.'+kind).show();
    },
  };
})();
