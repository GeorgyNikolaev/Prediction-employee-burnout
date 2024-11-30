$(function() {

    /* navToggal */
    let navToggal = $('#navToggal')
    let nav = $('#nav')

    navToggal.on('click', function(event) {
        event.preventDefault();

        $("body").toggleClass('show-nav')
        $(this).toggleClass('active')
        nav.toggleClass('show')
    });

    $(window).on("resize", function() {
        $("body").toggleClass('show-nav')
        headerScroll();
        navToggal.removeClass('active');
        nav.removeClass('show');
    });


    let intro = $("#intro");
    let header = $("#header");
    let introH = intro.innerHeight();
    let headerH = header.innerHeight();
    let scrollTop = $(window).scrollTop();

    /* Header scroll */

    headerScroll();

    $(window).on("scroll resize", function() {
        headerScroll();
    });


    function headerScroll() {
        introH = intro.innerHeight();
        headerH = header.innerHeight();

        let scrollTop = $(this).scrollTop();

        if( scrollTop >= (introH - headerH)) {
            header.addClass("header--dark");
        } else {
            header.removeClass("header--dark");
        }
    }

    /* smoove scroll */
    $("[data-scroll]").on("click", function(event) {
        event.preventDefault();

        let scrollEl = $(this).data("scroll");
        let scrollElPos = $(scrollEl).offset().top;

        // $("body").toggleClass('show-nav')
        $("body").removeClass('show-nav')
        navToggal.removeClass('active');
        nav.removeClass('show');


        $("html, body").animate({
            scrollTop: scrollElPos - headerH
        }, 500)
    });


    /*  scrollspy  */
    let windowH = $(window).height();

    scrollSpy(scrollTop);

    $(window).on("scroll", function() {
        scrollTop = $(this).scrollTop();

        scrollSpy(scrollTop);
    });

    function scrollSpy(scrollTop) {
        $("[data-scrollspy]").each(function() {
            let $this = $(this);
            let sectionId = $this.data('scrollspy');
            let sectionOffset = $this.offset().top;

            sectionOffset = sectionOffset - (windowH / 3);

            if(scrollTop >= sectionOffset) {
                $('#nav [data-scroll]').removeClass('active');
                $('#nav [data-scroll="' + sectionId + '"]').addClass('active');
            }

            if (scrollTop == 0) {
                $('#nav [data-scroll="' + sectionId + '"]').removeClass('active');
            }
        });
    }

    /* Modal */
    $('[data-modal]').on('click', function(event) {
        event.preventDefault();

        let modal = $(this).data('modal');

        $('body').addClass('no-scroll');
        $(modal).addClass('show');

        setTimeout(function() {
            $(modal).find('.modal__content').css({
                transform: 'translateY(0)',
                opacity: '1'
            });
        }, 100);
    });

    $('[data-modal-close]').on('click', function(event) {
        event.preventDefault();
        let modal = $(this).parents('.modal');

        modalCose(modal);
    });

    $('.modal').on('click', function() {
        let modal = $(this);

        modalCose(modal);
    });

    $('.modal__content').on('click', function() {
        event.stopPropagation();
    });


    function modalCose(modal ) {
        modal.find('.modal__content').css({
            transform: 'translateY(100px)',
            opacity: '0'
        });

        setTimeout(function() {
            $('body').removeClass('no-scroll');
            modal.removeClass('show');
        }, 100);
    }


    /* Slick slider https://kenwheeler.github.io/slick/
    Intro slider*/
    let introSlider = $("#introSlider");

    introSlider.slick({
        infinite: true,
        slidesToShow: 1,
        slidesToScroll: 1,
        arrows: false,
        fade: true,
        autoplay: true,
        autoplaySpeed: 4000,
        speed: 1000
    });

    $('#introSliderPrev').on('click', function() {
        introSlider.slick('slickPrev')
    });

    $('#introSliderNext').on('click', function() {
        introSlider.slick('slickNext')
    });

    /* reviewsslider */
    let reviewsSlider = $("#reviewsSlider");

    reviewsSlider.slick({
        infinite: true,
        slidesToShow: 1,
        slidesToScroll: 1,
        arrows: false,
        dots: true,
        speed: 500,
        autoplay: true,
        autoplaySpeed: 4000
    });

    //AOS
    // You can also pass an optional settings object
    // below listed default settings
    AOS.init({
      // Global settings:
      disable: 'mobile', // accepts following values: 'phone', 'tablet', 'mobile', boolean, expression or function
      startEvent: 'DOMContentLoaded', // name of the event dispatched on the document, that AOS should initialize on
      initClassName: 'aos-init', // class applied after initialization
      animatedClassName: 'aos-animate', // class applied on animation
      useClassNames: false, // if true, will add content of `data-aos` as classes on scroll
      disableMutationObserver: false, // disables automatic mutations' detections (advanced)
      debounceDelay: 50, // the delay on debounce used while resizing window (advanced)
      throttleDelay: 99, // the delay on throttle used while scrolling the page (advanced)


      // Settings that can be overridden on per-element basis, by `data-aos-*` attributes:
      offset: 120, // offset (in px) from the original trigger point
      delay: 0, // values from 0 to 3000, with step 50ms
      duration: 600, // values from 0 to 3000, with step 50ms
      easing: 'ease', // default easing for AOS animations
      once: false, // whether animation should happen only once - while scrolling down
      mirror: false, // whether elements should animate out while scrolling past them
      anchorPlacement: 'top-bottom', // defines which position of the element regarding to window should trigger the animation

    });



});


