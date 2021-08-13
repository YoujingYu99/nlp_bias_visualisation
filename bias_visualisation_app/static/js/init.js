(function($){
  $(function(){

    $('.sidenav').sidenav();
    $('.parallax').parallax();
    // TABS INIT
     $('.tabs').tabs();
      // CAROUSEL INIT
      $('.carousel.carousel-slider').carousel({ fullWidth: true });
         // SLIDER INIT
      $('.slider').slider({
        indicators: true,
        // we want the little dots to show
        height: 500,
        transition: 400,
        interval: 4000
        // how long the slide stays for
      })
      ;


  }); // end of document ready
})(jQuery); // end of jQuery name space


