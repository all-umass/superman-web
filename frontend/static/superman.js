// Global variables!
//  fig: figure number for the page's figure
//  upload_cbs: generic-ish callback functions {success: fn, fail: fn}
var fig, upload_cbs;

function multi_val(jq_list) {
  return jq_list.map(function(){ return this.value; }).toArray();
}

var GetArgs = (function(){
  return {
    plot: function(post_data) {
      post_data['alpha'] = $("#plt_alpha").val();
      post_data['legend'] = +$("#plt_legend").is(":checked");
      post_data['line_width'] = $("#plt_lw").val();
      post_data['cmap'] = $("#plt_cmap").val();
      return post_data;
    },
    resample: function(ctx, post_data) {
      var table = $('tbody', ctx);
      post_data['crop_lb'] = multi_val(table.find('.crop_lb'));
      post_data['crop_ub'] = multi_val(table.find('.crop_ub'));
      post_data['crop_step'] = multi_val(table.find('.crop_step'));
      return post_data;
    },
    baseline: function(ctx, post_data, method) {
      var table = $(ctx);
      if (method === undefined) {
        method = table.find('.blr_method').val();
      }
      post_data['blr_method'] = method;
      post_data['blr_segmented'] = table.find('.blr_segmented').is(':checked');
      post_data['blr_inverted'] = table.find('.blr_inverted').is(':checked');
      post_data['blr_flip'] = table.find('.blr_flip').is(':checked');
      if (method) {
        var idx = method.length + 1;
        table.find('.param.'+method+'>span').each(function(i,e){
          var param = e.className.substr(idx);
          post_data['blr_' + param] = e.innerHTML;
        });
      }
      return post_data;
    },
    pp: function(ctx) {
      return $(ctx).find('.pp_staging > li').map(function(){
        return $(this).text();
      }).toArray().join(',');
    },
  };
})();

var SingleSpectrum = (function(){
  function _make_post_callbacks(msg_selector) {
    return {
      success: function(data, status) {
        var msg = $(msg_selector);
        msg.text(msg.text() + " " + status + ".").delay(3000).fadeOut();
        if (status === 'success' && data.length == 4) {
          update_zoom_ctrl(data);
        }
      },
      fail: function(jqXHR, textStatus, errorThrown) {
        var msg = $(msg_selector);
        var out = msg.text() + " " + textStatus + ": " + jqXHR.responseText;
        msg.text(out);  // no delay, keep the error message up
      }
    };
  }

  var pp_cbs = _make_post_callbacks('#messages');
  var bl_cbs = _make_post_callbacks('#baseline_messages');
  // XXX: once peakfit doesn't need these, make this private
  upload_cbs = _make_post_callbacks('#upload_messages');

  return {
    upload: function(elt) {
      var err_span = $(elt).next('.err_msg').empty();
      if (elt.files.length != 1) {
        return err_span.text("Choose a file to upload");
      }
      var f = elt.files[0];
      if (f.size > 5000000) {
        return err_span.text("File must be <5mb");
      }
      $('#upload_messages').text("Uploading...").fadeIn();
      var post_data = new FormData();
      post_data.append('fignum', fig.id);
      post_data.append('query', f);
      $.ajax({
        url: '/_upload_spectrum',
        data: post_data,
        processData: false,
        contentType: false,
        dataType: 'json',
        type: 'POST',
        error: upload_cbs['fail'],
        success: function (data, status) {
          upload_cbs['success'](data, status);
          elt.value = "";  // reset the input
        }
      });
    },
    select: function(name_kind) {
      if (name_kind == '') return;
      var parts = name_kind.split(','),
          post_data = {name: parts[0], kind: parts[1], fignum: fig.id},
          spinner = $('#spinner').show(),
          selector = $('#selector').empty();
      function do_select(name, idx) {
        $('#upload_messages').text("Selecting...").fadeIn();
        $.ajax({
          url: '/_select', type: 'POST', dataType: 'json',
          data: {name: name, idx: idx, ds_name: parts[0], ds_kind: parts[1],
                 fignum: fig.id},
          error: upload_cbs['fail'], success: upload_cbs['success']
        });
      }
      selector.load('/_spectrum_selector', post_data, function(){
        spinner.hide();
        // If we have a numeric spinner
        $('input', selector).change(function(evt){
          do_select(undefined, evt.target.value)
        });
        // If we have a fancy <select> dropdown
        $('select', selector).select2().change(function(evt){
          if (evt.target.value.length > 0) {
            do_select(evt.target.value, undefined);
          }
        });
        // toggle .libs_only elements
        $('.libs_only').toggle(parts[1] === 'LIBS');
      });
    },
    preprocess: function(ctx) {
      var post_data = { pp: GetArgs.pp(ctx), fignum: fig.id };
      $('#messages').text("Preprocessing...").fadeIn();
      $.post('/_pp', post_data, pp_cbs['success'], 'json').fail(pp_cbs['fail']);
    },
    baseline: function(ctx, method) {
      var post_data = GetArgs.baseline(ctx, {fignum: fig.id}, method);
      GetArgs.resample(ctx, post_data);
      $('#baseline_messages').text("Correcting baseline...").fadeIn();
      $.post('/_baseline', post_data, bl_cbs['success']).fail(bl_cbs['fail']);
    },
  };
})();

function do_filter(filter_element, post_data) {
  var elt = $(filter_element);
  elt.prop('disabled', true);
  elt.children('.ing').text("ing...").fadeIn();
  post_data['fignum'] = fig.id;

  $.post('_filter', post_data, function(data, status) {
    elt.prop('disabled', false);
    if (status === 'success') {
      var num_spectra = parseInt(data);
      if (num_spectra <= 99999) {
        $('.needs_filter').prop('disabled', false);
      }
      elt.children('.ing').text(': ' + num_spectra);
    }
  }, 'json');
}
function update_1eX(val, selector, show_e) {
  var x = Math.pow(10, val);
  $(selector).text(show_e ? x.toExponential(3) : x.toFixed(3));
}
function onready_boilerplate(ws_uri, fignum) {
  function ondownload(figure, format) {
    window.open('/'+figure.id+'/download.' + format, '_blank');
  }
  $('body').on('contextmenu', '#figure', function(e){ return false; });
  var fig_div = $('div#figure'),
      fig_width = fig_div.width(),
      fig_height = fig_div.height(),
      em_px = parseFloat(fig_div.css('font-size')),
      canvas_width = fig_width - 4,
      canvas_height = fig_height - 1.8 * em_px;
  var websocket_type = mpl.get_websocket_type();
  var websocket = new websocket_type(ws_uri + fignum + "/ws");
  fig = new mpl.figure(fignum, websocket, ondownload, fig_div);
  // Hack to disable pgf image download option
  $('.mpl-toolbar-option option')[2].disabled = true;
  // wait for the websocket to be ready, then ask the figure to resize
  var check_ready = setInterval(function(){
    if (websocket.readyState === 1) {
      clearInterval(check_ready);
      fig.request_resize(canvas_width, canvas_height);
      // lock in the current size for fig_div
      fig_div.css({width: fig_width, 'min-height': fig_height});
    }
  }, 100);
}

function update_zoom_ctrl(data) {
  var zc = $('#zoom_control');
  $('input[name=xmin]', zc).val(data[0].toPrecision(6));
  $('input[name=xmax]', zc).val(data[1].toPrecision(6));
  $('input[name=ymin]', zc).val(data[2].toPrecision(6));
  $('input[name=ymax]', zc).val(data[3].toPrecision(6));
  $('.needs_plot').attr('disabled', false);
}
function do_zoom() {
  var zc = $('#zoom_control');
  $.post('/_zoom', {
    xmin: $('input[name=xmin]', zc).val(),
    xmax: $('input[name=xmax]', zc).val(),
    ymin: $('input[name=ymin]', zc).val(),
    ymax: $('input[name=ymax]', zc).val(),
    fignum: fig.id
  });
}
function add_pp_step(ctx, name, step, input_name) {
  var table = $(ctx).closest('table');
  var parts = [name];
  if (step !== null) {
    parts.push(step);
  }
  if (input_name !== null) {
    var arg_inputs = table.find('input[name="'+input_name+'"]');
    parts = parts.concat(arg_inputs.map(function(){
      return this.value;
    }).toArray());
  }
  if (parts.length < 2) return;
  table.find('.pp_staging').append(
    '<li class="'+name+'" onclick="$(this).remove()">' +
    parts.join(':') + '</li>');
}
