$(function(){

   /* 
    var optarray = $("#requests").children('option').map(function() {
        return {
            "value": this.value,
            "id":this.id,
            "option": "<option value='" + this.value + "' id=" + this.id + ">" + this.text + "</option>"
        }
    });
        
    $("#tasks").change(function() {
        $("#requests").children('option').remove();
        var addoptarr = [];
        for (i = 0; i < optarray.length; i++) {
            if (optarray[i].value.indexOf($(this).val()) > -1) {
                addoptarr.push(optarray[i].option);
            }
        }
        $("#requests").html(addoptarr.join(''))
    }).change();
    //})
    */

    $("#tasks").change(function(){
        var temp = document.querySelector("#tasks");
        var value = temp.options[temp.selectedIndex].value;
        $.get('/requests', {value:value})
        .done(function(data){
            $("#requests").html("");
            
            for (i=0;i<data['len_reqs'];i++){
                //$('#requests').append("<option SELECTED> hello</option");
                $("#requests").append("<option value= " + data['value'][i] + " id=" + data['req_id'][i] + " SELECTED>"+data['req_text'][i]+ "</option>");
            }
        });

    });

    $("#requests").change(function(){
    //$(document).ready(function() {
        var dnd = document.querySelector('#requests');
        var r_id = dnd.options[dnd.selectedIndex].id;
        $.get("/info", {req_id: r_id})
        .done(function(data) {
            var req_id = data['req_id'];
            var task_title = data['task_title'];
            var task_stmt = data['task_stmt'];
            var task_narr = data['task_narr'];
            var task_in_scope = data['task_in_scope'];
            var task_not_in_scope = data['task_not_in_scope'];
            var req_text = data['req_text'];

            $("#req_id").text(req_id);
            $("#title").text("Task title: "+task_title);
            $("#statement").text("Task statement: "+task_stmt);
            $("#narrative").text("Task narrative: "+task_narr);
            $("#scope").text("In scope: "+task_in_scope);
            $("#out_scope").text("Not in scope: "+task_not_in_scope);
            $("#request").text("Request: "+req_text);
            //$("#place_for_info").append("<a href={{ url_for('terms', req_id=" + req_id + ")}}>Edit Keywords</a>");
            //$("#place_for_info").append('<a href="{{ url_for(\'terms\', req_id='+ req_id + ') }}">Edit Keywords</a>')


        });

    });

    $("#requests").change(function(){
        var dnd = document.querySelector('#requests');
        var r_id = dnd.options[dnd.selectedIndex].id;

        if(document.getElementById('tm').checked) {
            var flag = 1;
        } else {
            var flag = 0;
        }

        if(document.getElementById('al').checked) {
            var al_flag = 1;
        } else {
            var al_flag = 0;
        }

        if(document.getElementById('tm').checked) {
            $.get("/topic", {req_id:r_id})
            .done(function(response){
             $("#place_for_documents").html(response);
            });

        // to manually do it without the template
        //    $.get("/topic", {req_id:r_id})
        //    .done(function(data){
        //        var snippets = data["snippets"];
        //        var topics = data["topics"];
        //        //$("#snippets").append(snippets)
        //        $("#place_for_documents").html("");
        //        $("#place_for_documents").append('<a><b>Document cluster label: </b>' + topics + "</a><br></br>");
        //        for (let i = 0; i < 50; i++){
        //            $("#place_for_documents").append(snippets[i][1] + "<hr>");
                    //$("#place_for_documents").append("<p>"+snippets[i]+"<p>");
        //        }
        //    });

        } else {
            $.get("/docu", {req_id:r_id, flag:flag, al_flag:al_flag})
            .done(function(data){
                var snippets = data["snippets"];
                //$("#snippets").append(snippets)
                $("#place_for_documents").html("");
                for (let i = 0; i < 50; i++){
                    $("#place_for_documents").append('<button id = '+snippets[i][0]+' class=snips>Document ' + snippets[i][0]+"</button><br></br>");
                    $("#place_for_documents").append(snippets[i][1] + "<hr>");
                    //$("#place_for_documents").append("<p>"+snippets[i]+"<p>");
                }
            });
        }
    });

   // $("#tasks").change(function(){
        //var dnd = document.querySelector('#tasks');
        //var r_id = dnd.options[dnd.selectedIndex].id;
   //     $.get("/request").done(function(data){
   //         var text = data['req_text']
   //         $('#requests').html("");
   //         for (i =0; i < 2; i++){
   //             $("#requests").append('<option value= "{{req_text}}" SELECTED>'+ text[0][i] + '</option>');
   //         }
   //     });
        // $("#requests").append(<option value= "{{req_text}}" SELECTED>{{req_text}}</option>)
        

   // });

    $(document).on('click', '.snips', function () {
        var id = $(this).attr('id');
        var dnd = document.querySelector('#requests');
        var r_id = dnd.options[dnd.selectedIndex].id;

       $.get("/ui", {req_id:r_id, doc_id:id})
       .done(function(response){
        $("#place_for_documents").html(response);
       });

    });

    $('#test').click(function (){
        var dnd = document.querySelector('#requests');
        var r_id = dnd.options[dnd.selectedIndex].id;
        if(document.getElementById('tm').checked) {
            var flag = 1;
        } else {
            var flag = 0;
        }

        $.get("/docu", {req_id:r_id, flag:flag})
        .done(function(data){
            var snippets = data["snippets"];
            //$("#snippets").append(snippets)
            $("#place_for_documents").html("")
            for (let i = 0; i < 50; i++){
                $("#place_for_documents").append('<button id = '+snippets[i][0]+' class=snips>Document ' + snippets[i][0]+"</button><br></br>");
                $("#place_for_documents").append(snippets[i][1] + "<hr>");
                //$("#place_for_documents").append("<p>"+snippets[i]+"<p>");
            }
        });

        //alert("testinggg");
    });

    $('#save_sub').click(function(){
        var dnd = document.querySelector('#requests');
        var r_id = dnd.options[dnd.selectedIndex].id;
        if(document.getElementById('tm').checked) {
            var flag = 1;
        } else {
            var flag = 0;
        }

        $.get("/docu", {req_id:r_id, flag:flag})
        .done(function(data){
            var snippets = data["snippets"];
            //$("#snippets").append(snippets)
            $("#place_for_documents").html("")
            for (let i = 0; i < 50; i++){
                $("#place_for_documents").append('<button id = '+snippets[i][0]+' class=snips>Document ' + snippets[i][0]+"</button><br></br>");
                $("#place_for_documents").append(snippets[i][1] + "<hr>");
                //$("#place_for_documents").append("<p>"+snippets[i]+"<p>");
            }
        });

    });
    //$(document).on('click', '.btn btn-secondary', function (){
    //    alert("test");
    //});

    //$('.snippets').find('a[href="#"]').on('click', function (e) {
    //    e.preventDefault();
    //    this.expand = !this.expand;
    //    $(this).text(this.expand?"Click to collapse":"Click to read more");
    //    $(this).closest('.snippets').find('.small, .big').toggleClass('small big');
    //});


    $(".keyword").on("click", function() {
        var dnd = document.querySelector('#requests');
        var r_id = dnd.options[dnd.selectedIndex].id;

        $.ajax({
            type: "GET",
            url: "/terms",
            data: {req_id:r_id},
            success:function(response){
                document.write(response); 
           }
    
        });
        //flask_util.url_for("term",req_id=r_id)
     /*
        $.ajax({
            'url' : '/terms',
            'type' : 'POST',
            'data' : {req_id:r_id},
            'success' : function(data) {   
            },
            'error' : function(request,error){
            }
        });
        */

    });

    $("#submit").click(function(){
        //var txt = document.getElementById('submit').value;
        var num = $(this).value;
        var txt = $("#fname").val();
        var dnd = document.querySelector('#requests');
        var r_id = dnd.options[dnd.selectedIndex].id; 

        $.ajax({
            type: "GET",
            url: "/update",
            data: {req_id:r_id, topic:txt, num:num},
            success:function(response){
                $("#place_for_documents").html(response);           
            }

        });

    });


    //$("#fname").keyup(function() {
    //    alert($(this).val());
    //});
});