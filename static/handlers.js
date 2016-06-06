// Global variables!
//  fig: figure number for the page's figure
//  ds_name: name of the selected dataset ('RRUFF', 'IRUG', etc.)
//  ds_kind: kind of the selected dataset ('Raman', 'LIBS', 'FTIR')
var fig, ds_name, ds_kind;

function update_zoom_ctrl(data) {
  $('#zoom_control input[name=xmin]').val(data[0]);
  $('#zoom_control input[name=xmax]').val(data[1]);
  $('#zoom_control input[name=ymin]').val(data[2]);
  $('#zoom_control input[name=ymax]').val(data[3]);
  $('.needs_plot').prop('disabled', false);
}
function make_post_callbacks(msg_selector) {
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
      var out = msg.text() + " " + textStatus + ". " + errorThrown;
      msg.text(out);  // no delay, keep the error message up
    }
  };
}
var upload_cbs = make_post_callbacks('#upload_messages');
function do_upload(elt) {
  if (elt.files.length != 1) {
    $('#upload_messages').text("Choose a file first!").fadeIn();
    return
  }
  var f = elt.files[0];
  if (f.size > 5000000) {
    $('#upload_messages').text("File must be <5mb").fadeIn();
    return
  }
  $('#upload_messages').text("Uploading...").fadeIn();
  var post_data = new FormData();
  post_data.append('fignum', fig.id);
  post_data.append('query', f);
  $.ajax({
    url: '/_upload',
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
}
function get_dataset(info) {
  var parts = info.split(',');
  ds_name = parts[0];
  ds_kind = parts[1];
  var post_data = {
    name: ds_name, kind: ds_kind, fignum: fig.id
  };
  function do_select(name, idx) {
    $('#upload_messages').text("Selecting...").fadeIn();
    $.post('/_select', {
      name: name, idx: idx, ds_name: ds_name, ds_kind: ds_kind, fignum: fig.id
    }, upload_cbs['success'], 'json').fail(upload_cbs['fail']);
  }
  $('#spinner').show();
  $('#selector').load('/_dataset_selector', post_data, function(){
    $('#spinner').hide();
    // If we have a numeric spinner
    $("#selector input").change(function(evt){
      do_select(undefined, evt.target.value)
    });
    // If we have a chosen <select> dropdown
    $("#selector .chosen-select").chosen({search_contains: true}).change(
    function(evt){
      if (evt.target.value.length > 0) {
        do_select(evt.target.value, undefined);
      }
    }).next().css('width', '+=15');
  });
}
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
        $('#plot_button').prop('disabled', false);
      }
      elt.children('.ing').text(': ' + num_spectra);
    }
  }, 'json');
}
function do_pp(pp) {
  $('#messages').text("Preprocessing...").fadeIn();
  var post_data = { pp: pp, fignum: fig.id };
  var cbs = make_post_callbacks('#messages');
  $.post('/_pp', post_data, cbs['success'], 'json').fail(cbs['fail']);
}
function do_baseline(method) {
  var msg = $('#baseline_messages');
  msg.text("Correcting baseline...").fadeIn();
  var post_data = add_baseline_args({fignum: fig.id}, method);
  var cbs = make_post_callbacks('#baseline_messages');
  $.post('/_baseline', post_data, cbs['success']).fail(cbs['fail']);
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
  var websocket_type = mpl.get_websocket_type();
  var websocket = new websocket_type(ws_uri + fignum + "/ws");
  fig = new mpl.figure(fignum, websocket, ondownload, $('div#figure'));
  $('.mpl-toolbar-option option')[2].disabled = true;  // Hack to disable pgf
}
function do_zoom() {
  var post_data = {
    xmin: $('#zoom_control input[name=xmin]').val(),
    xmax: $('#zoom_control input[name=xmax]').val(),
    ymin: $('#zoom_control input[name=ymin]').val(),
    ymax: $('#zoom_control input[name=ymax]').val(),
    fignum: fig.id
  };
  $.post('/_zoom', post_data);
}
function add_pp_step(ctx, name, step, input_name) {
  var table = $(ctx).closest('table');
  if (step === null) {
    step = table.find('input[name="'+input_name+'"]').map(function(){
      return this.value;
    }).toArray().join(':');
  }
  if (step === "") return;
  table.find('.pp_staging').append(
    '<li class="'+name+'" onclick="$(this).remove()">' +
    name + ':' + step + '</li>');
}
function collect_pp_args(ctx) {
  return $(ctx).find('.pp_staging > li').map(function(){
    return $(this).text();
  }).toArray().join(',');
}
function add_baseline_args(post_data, method) {
  if (method === undefined) {
    method = $('#blr_method').val();
  }
  if (!method) return post_data;
  post_data['blr_method'] = method;
  post_data['blr_segmented'] = $('#blr_segmented').is(':checked');
  post_data['blr_inverted'] = $('#blr_inverted').is(':checked');
  post_data['blr_lb'] = $('#blr_lb').val();
  post_data['blr_ub'] = $('#blr_ub').val();
  var idx = method.length + 1;
  $('td.param.'+method+'>span').each(function(i,e){
    var param = e.id.substr(idx);
    post_data['blr_' + param] = e.innerHTML;
  });
  return post_data;
}
